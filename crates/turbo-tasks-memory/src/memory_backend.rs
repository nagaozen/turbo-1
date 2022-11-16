use std::{
    borrow::Cow,
    cell::RefCell,
    cmp::min,
    collections::{hash_map, HashMap, VecDeque},
    future::Future,
    hash::BuildHasherDefault,
    pin::Pin,
    time::{Duration, Instant},
};

use anyhow::{bail, Result};
use auto_hash_map::AutoSet;
use concurrent_queue::ConcurrentQueue;
use dashmap::{mapref::entry::Entry, DashMap};
use rustc_hash::FxHasher;
use tokio::task::futures::TaskLocalFuture;
use turbo_tasks::{
    backend::{
        Backend, BackendJobId, CellContent, PersistentTaskType, TaskExecutionSpec,
        TransientTaskType,
    },
    event::EventListener,
    util::{IdFactory, NoMoveVec},
    CellId, RawVc, TaskId, TraitTypeId, TurboTasksBackendApi,
};

use crate::{
    cell::RecomputingCell,
    gc::{GcAction, GcItem},
    output::Output,
    scope::{TaskScope, TaskScopeId},
    task::{
        run_add_to_scope_queue, run_remove_from_scope_queue, Task, TaskDependency,
        DEPENDENCIES_TO_TRACK,
    },
};

pub struct MemoryBackend {
    memory_tasks: NoMoveVec<Task, 13>,
    memory_task_scopes: NoMoveVec<TaskScope>,
    scope_id_factory: IdFactory<TaskScopeId>,
    pub(crate) initial_scope: TaskScopeId,
    backend_jobs: NoMoveVec<Job>,
    backend_job_id_factory: IdFactory<BackendJobId>,
    task_cache: DashMap<PersistentTaskType, TaskId, BuildHasherDefault<FxHasher>>,
    gc_queue: ConcurrentQueue<TaskId>,
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBackend {
    pub fn new() -> Self {
        let memory_task_scopes = NoMoveVec::new();
        let scope_id_factory = IdFactory::new();
        let initial_scope: TaskScopeId = scope_id_factory.get();
        unsafe {
            memory_task_scopes.insert(*initial_scope, TaskScope::new_active(initial_scope, 0, 0));
        }
        Self {
            memory_tasks: NoMoveVec::new(),
            memory_task_scopes,
            scope_id_factory,
            initial_scope,
            backend_jobs: NoMoveVec::new(),
            backend_job_id_factory: IdFactory::new(),
            task_cache: DashMap::default(),
            gc_queue: ConcurrentQueue::unbounded(),
        }
    }

    fn connect_task_child(
        &self,
        parent: TaskId,
        child: TaskId,
        reason: &'static str,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        self.with_task(parent, |parent| {
            parent.connect_child(child, reason, self, turbo_tasks)
        });
    }

    pub(crate) fn create_backend_job(&self, job: Job) -> BackendJobId {
        let id = self.backend_job_id_factory.get();
        // SAFETY: This is a fresh id
        unsafe {
            self.backend_jobs.insert(*id, job);
        }
        id
    }

    fn try_get_output<T, F: FnOnce(&mut Output) -> Result<T>>(
        &self,
        id: TaskId,
        strongly_consistent: bool,
        note: impl Fn() -> String + Sync + Send + 'static,
        turbo_tasks: &dyn TurboTasksBackendApi,
        func: F,
    ) -> Result<Result<T, EventListener>> {
        self.with_task(id, |task| {
            task.get_or_wait_output(strongly_consistent, func, note, self, turbo_tasks)
        })
    }

    pub fn with_all_cached_tasks(&self, mut func: impl FnMut(TaskId)) {
        for id in self.task_cache.clone().into_read_only().values() {
            func(*id);
        }
    }

    pub fn with_task<T>(&self, id: TaskId, func: impl FnOnce(&Task) -> T) -> T {
        func(self.memory_tasks.get(*id).unwrap())
    }

    pub fn with_scope<T>(&self, id: TaskScopeId, func: impl FnOnce(&TaskScope) -> T) -> T {
        func(self.memory_task_scopes.get(*id).unwrap())
    }

    pub fn create_new_scope(&self, tasks: usize) -> TaskScopeId {
        let id = self.scope_id_factory.get();
        unsafe {
            self.memory_task_scopes
                .insert(*id, TaskScope::new(id, tasks));
        }
        id
    }

    fn increase_scope_active_queue(
        &self,
        mut queue: Vec<TaskScopeId>,
        reason: &'static str,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        while let Some(scope) = queue.pop() {
            if let Some(tasks) = self.with_scope(scope, |scope| {
                scope.state.lock().increment_active(&mut queue)
            }) {
                turbo_tasks.schedule_backend_foreground_job(
                    self.create_backend_job(Job::ScheduleWhenDirtyFromScope(tasks, reason)),
                );
            }
        }
    }

    pub(crate) fn increase_scope_active(
        &self,
        scope: TaskScopeId,
        reason: &'static str,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        self.increase_scope_active_queue(vec![scope], reason, turbo_tasks);
    }

    pub(crate) fn increase_scope_active_by(
        &self,
        scope: TaskScopeId,
        count: usize,
        reason: &'static str,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        let mut queue = Vec::new();
        if let Some(tasks) = self.with_scope(scope, |scope| {
            scope.state.lock().increment_active_by(count, &mut queue)
        }) {
            turbo_tasks.schedule_backend_foreground_job(
                self.create_backend_job(Job::ScheduleWhenDirtyFromScope(tasks, reason)),
            );
        }
        self.increase_scope_active_queue(queue, reason, turbo_tasks);
    }

    pub(crate) fn decrease_scope_active(
        &self,
        scope: TaskScopeId,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        self.decrease_scope_active_by(scope, 1, turbo_tasks);
    }

    pub(crate) fn decrease_scope_active_by(
        &self,
        scope: TaskScopeId,
        count: usize,
        _turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        let mut queue = vec![scope];
        while let Some(scope) = queue.pop() {
            self.with_scope(scope, |scope| {
                scope.state.lock().decrement_active_by(count, &mut queue)
            });
        }
    }

    fn add_task_to_gc(&self, task: TaskId) {
        self.gc_queue.push(task).unwrap();
    }

    pub fn run_gc(&self, turbo_tasks: &dyn TurboTasksBackendApi) {
        const MAX_TASKS_TO_CHECK: usize = 1000000;
        const INCREMENT: usize = 100000;
        const MAX_COLLECT_PERCENTAGE: usize = 30;
        const MB: usize = 1024 * 1024;
        #[cfg(not(debug_assertions))]
        const GB: usize = 1024 * MB;

        #[cfg(debug_assertions)]
        const LOWER_MEM_TARGET: usize = 300 * MB;
        #[cfg(debug_assertions)]
        const UPPER_MEM_TARGET: usize = 1000 * MB;
        #[cfg(debug_assertions)]
        const MEM_LIMIT: usize = 2000 * MB;

        #[cfg(not(debug_assertions))]
        const LOWER_MEM_TARGET: usize = 3 * GB;
        #[cfg(not(debug_assertions))]
        const UPPER_MEM_TARGET: usize = 4 * GB;
        #[cfg(not(debug_assertions))]
        const MEM_LIMIT: usize = 8 * GB;

        let usage = turbo_malloc::TurboMalloc::memory_usage();

        if usage < UPPER_MEM_TARGET {
            return;
        }

        let mut tasks = HashMap::with_capacity(INCREMENT);
        let now = Instant::now();
        let mut i = 0;
        while tasks.len() < INCREMENT && i < MAX_TASKS_TO_CHECK {
            if let Ok(id) = self.gc_queue.pop() {
                i += 1;
                if let Some(info) = self.with_task(id, |task| task.gc_info(now, self)) {
                    tasks.insert(id, info);
                } else {
                    // Put task back into the queue
                    let _ = self.gc_queue.push(id);
                }
            } else {
                break;
            }
        }
        let mut compute_duration_map = HashMap::new();
        for (id, info) in tasks.iter() {
            compute_duration_map.insert(*id, info.compute_duration);
        }

        let mut items = Vec::new();
        for (&id, info) in tasks.iter_mut() {
            let mut all_tasks = HashMap::new();
            if info.unread_cells > 0 {
                items.push(GcItem {
                    action: GcAction::UnreadCells(id),
                    compute_duration: info.compute_duration,
                    age: info.age,
                });
            }
            for (i, tasks) in info.cells.drain(..) {
                items.push(GcItem {
                    action: GcAction::ReadCell(id, i),
                    compute_duration: info.compute_duration
                        + tasks
                            .into_iter()
                            .map(|id| match all_tasks.entry(id) {
                                hash_map::Entry::Occupied(e) => *e.get(),
                                hash_map::Entry::Vacant(e) => {
                                    *e.insert(*compute_duration_map.entry(id).or_insert_with(
                                        || self.with_task(id, |task| task.gc_compute_duration()),
                                    ))
                                }
                            })
                            .sum(),
                    age: info.age,
                });
            }
            info.output.drain(..).for_each(|id| {
                if let hash_map::Entry::Vacant(e) = all_tasks.entry(id) {
                    let duration = *compute_duration_map
                        .entry(id)
                        .or_insert_with(|| self.with_task(id, |task| task.gc_compute_duration()));
                    e.insert(duration);
                }
            });
            if !info.active {
                items.push(GcItem {
                    action: GcAction::Unload(id),
                    compute_duration: info.compute_duration + all_tasks.into_values().sum(),
                    age: info.age,
                });
            }
        }
        items.sort_by(GcItem::cmp_priority);
        let len = items.len();
        let collect_count =
            len * min(
                MAX_COLLECT_PERCENTAGE,
                (usage - LOWER_MEM_TARGET) * MAX_COLLECT_PERCENTAGE
                    / (MEM_LIMIT - LOWER_MEM_TARGET),
            ) / 100;
        items.truncate(collect_count);
        let collect_count = items.len();

        let mut collected_tasks = 0;
        if !items.is_empty() {
            items.sort_by(GcItem::cmp_task);
            let mut current_task = items[0].task();
            let mut current_unload = false;
            let mut current_unread = false;
            let mut current_cells = Vec::new();
            macro_rules! gc_current {
                () => {
                    self.with_task(current_task, |task| {
                        if current_unload {
                            if task.unload(self, turbo_tasks) {
                                return;
                            }
                        }
                        task.gc(current_unread, current_cells);
                    });
                };
            }
            for item in items {
                if item.task() != current_task {
                    collected_tasks += 1;
                    gc_current!();
                    current_task = item.task();
                    current_unread = false;
                    current_unload = false;
                    current_cells = Vec::new();
                }
                match item.action {
                    GcAction::UnreadCells(_) => current_unread = true,
                    GcAction::ReadCell(_, i) => current_cells.push(i),
                    GcAction::Unload(_) => current_unload = true,
                }
            }
            collected_tasks += 1;
            gc_current!();
        }
        let inspected_task_count = tasks.len();
        for (id, _) in tasks {
            let _ = self.gc_queue.push(id);
        }

        turbo_malloc::TurboMalloc::collect();
        let new_usage = turbo_malloc::TurboMalloc::memory_usage();
        println!(
            "GC collected {}/{} items of {}/{} tasks  {:.3} GB -> {:.3} GB ({} tasks in queue)",
            collect_count,
            len,
            collected_tasks,
            inspected_task_count,
            (usage / 1000_000) as f32 / 1000.0,
            (new_usage / 1000_000) as f32 / 1000.0,
            self.gc_queue.len()
        );

        // println!(
        //     "memory_tasks: {} entries capacity {} kB",
        //     self.memory_tasks.capacity(),
        //     std::mem::size_of::<Task>() * self.memory_tasks.capacity() / 1024
        // );
        // println!(
        //     "memory_task_scopes: {} entries capacity {} kB",
        //     self.memory_task_scopes.capacity(),
        //     std::mem::size_of::<TaskScope>() * self.memory_task_scopes.capacity() /
        // 1024 );
        // println!(
        //     "backend_jobs: {} entries capacity {} kB",
        //     self.backend_jobs.capacity(),
        //     std::mem::size_of::<Job>() * self.backend_jobs.capacity() / 1024
        // );

        // if inspected_task_count > 0 {
        let job = self.create_backend_job(Job::GarbaggeCollection);
        turbo_tasks.schedule_backend_background_job(job);
        // }
    }
}

impl Backend for MemoryBackend {
    fn idle_start(&self, turbo_tasks: &dyn TurboTasksBackendApi) {
        // let job = self.create_backend_job(Job::GarbaggeCollection);
        // turbo_tasks.schedule_backend_background_job(job);
    }

    fn invalidate_task(
        &self,
        task: TaskId,
        reason: &'static str,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        self.with_task(task, |task| task.invalidate(reason, self, turbo_tasks));
    }

    fn invalidate_tasks(
        &self,
        tasks: Vec<TaskId>,
        reason: &'static str,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        for task in tasks.into_iter() {
            self.with_task(task, |task| {
                task.invalidate(reason, self, turbo_tasks);
            });
        }
    }

    fn get_task_description(&self, task: TaskId) -> String {
        self.with_task(task, |task| task.get_description())
    }

    type ExecutionScopeFuture<T: Future<Output = Result<()>> + Send + 'static> =
        TaskLocalFuture<RefCell<AutoSet<TaskDependency>>, T>;
    fn execution_scope<T: Future<Output = Result<()>> + Send + 'static>(
        &self,
        _task: TaskId,
        future: T,
    ) -> Self::ExecutionScopeFuture<T> {
        DEPENDENCIES_TO_TRACK.scope(Default::default(), future)
    }

    fn try_start_task_execution(
        &self,
        task: TaskId,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) -> Option<TaskExecutionSpec> {
        self.with_task(task, |task| {
            if task.execution_started(self, turbo_tasks) {
                Some(TaskExecutionSpec {
                    future: task.execute(turbo_tasks),
                })
            } else {
                None
            }
        })
    }

    fn task_execution_result(
        &self,
        task: TaskId,
        result: Result<Result<RawVc>, Option<Cow<'static, str>>>,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        self.with_task(task, |task| {
            task.execution_result(result, turbo_tasks);
        })
    }

    fn task_execution_completed(
        &self,
        task_id: TaskId,
        duration: Duration,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) -> bool {
        self.with_task(task_id, |task| {
            task.execution_completed(duration, self, turbo_tasks)
        })
    }

    fn try_read_task_output(
        &self,
        task: TaskId,
        reader: TaskId,
        strongly_consistent: bool,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) -> Result<Result<RawVc, EventListener>> {
        if task == reader {
            bail!("reading it's own output is not possible");
        }
        self.try_get_output(
            task,
            strongly_consistent,
            move || format!("reading task output from {reader}"),
            turbo_tasks,
            |output| {
                Task::add_dependency_to_current(TaskDependency::TaskOutput(task));
                output.read(reader)
            },
        )
    }

    fn try_read_task_output_untracked(
        &self,
        task: TaskId,
        strongly_consistent: bool,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) -> Result<Result<RawVc, EventListener>> {
        self.try_get_output(
            task,
            strongly_consistent,
            || format!("reading task output untracked"),
            turbo_tasks,
            |output| output.read_untracked(),
        )
    }

    fn track_read_task_output(
        &self,
        task: TaskId,
        reader: TaskId,
        _turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        if task != reader {
            self.with_task(task, |t| {
                t.with_output_mut(|output| {
                    Task::add_dependency_to_current(TaskDependency::TaskOutput(task));
                    output.track_read(reader);
                })
            })
        }
    }

    fn try_read_task_cell(
        &self,
        task_id: TaskId,
        index: CellId,
        reader: TaskId,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) -> Result<Result<CellContent, EventListener>> {
        if task_id == reader {
            Ok(Ok(self.with_task(task_id, |task| {
                task.with_cell(index, |cell| cell.read_own_content_untracked())
            })))
        } else {
            Task::add_dependency_to_current(TaskDependency::TaskCell(task_id, index));
            self.with_task(task_id, |task| {
                match task.with_cell_mut(index, |cell| {
                    cell.read_content(
                        reader,
                        move || format!("{task_id} {index}"),
                        move || format!("reading {} {} from {}", task_id, index, reader),
                    )
                }) {
                    Ok(content) => Ok(Ok(content)),
                    Err(RecomputingCell { listener, schedule }) => {
                        if schedule {
                            task.invalidate(
                                "need garbagged collected data again",
                                self,
                                turbo_tasks,
                            );
                        }
                        Ok(Err(listener))
                    }
                }
            })
        }
    }

    fn try_read_own_task_cell_untracked(
        &self,
        current_task: TaskId,
        index: CellId,
        _turbo_tasks: &dyn TurboTasksBackendApi,
    ) -> Result<CellContent> {
        Ok(self.with_task(current_task, |task| {
            task.with_cell(index, |cell| cell.read_own_content_untracked())
        }))
    }

    fn try_read_task_cell_untracked(
        &self,
        task_id: TaskId,
        index: CellId,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) -> Result<Result<CellContent, EventListener>> {
        self.with_task(task_id, |task| {
            match task.with_cell_mut(index, |cell| {
                cell.read_content_untracked(
                    move || format!("{task_id}"),
                    move || format!("reading {} {} untracked", task_id, index),
                )
            }) {
                Ok(content) => Ok(Ok(content)),
                Err(RecomputingCell { listener, schedule }) => {
                    println!("recomputing cell {task_id} {index} untracked");
                    if schedule {
                        task.invalidate("need garbagged collected data again", self, turbo_tasks);
                    }
                    Ok(Err(listener))
                }
            }
        })
    }

    fn track_read_task_cell(
        &self,
        task: TaskId,
        index: CellId,
        reader: TaskId,
        _turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        if task != reader {
            Task::add_dependency_to_current(TaskDependency::TaskCell(task, index));
            self.with_task(task, |task| {
                task.with_cell_mut(index, |cell| cell.track_read(reader))
            });
        }
    }

    fn try_read_task_collectibles(
        &self,
        id: TaskId,
        trait_id: TraitTypeId,
        reader: TaskId,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) -> Result<Result<AutoSet<RawVc>, EventListener>> {
        self.with_task(id, |task| {
            task.try_read_task_collectibles(reader, trait_id, self, turbo_tasks)
        })
    }

    fn emit_collectible(
        &self,
        trait_type: TraitTypeId,
        collectible: RawVc,
        id: TaskId,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        self.with_task(id, |task| {
            task.emit_collectible(trait_type, collectible, self, turbo_tasks)
        });
    }

    fn unemit_collectible(
        &self,
        trait_type: TraitTypeId,
        collectible: RawVc,
        id: TaskId,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        self.with_task(id, |task| {
            task.unemit_collectible(trait_type, collectible, self, turbo_tasks)
        });
    }

    fn update_task_cell(
        &self,
        task: TaskId,
        index: CellId,
        content: CellContent,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) {
        self.with_task(task, |task| {
            task.with_cell_mut(index, |cell| cell.assign(content, turbo_tasks))
        })
    }

    /// SAFETY: Must only called once with the same id
    fn run_backend_job<'a>(
        &'a self,
        id: BackendJobId,
        turbo_tasks: &'a dyn TurboTasksBackendApi,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        // SAFETY: id will not be reused until with job is done
        if let Some(job) = unsafe { self.backend_jobs.take(*id) } {
            Box::pin(async move {
                job.run(self, turbo_tasks).await;
                // SAFETY: This id will no longer be used
                unsafe {
                    self.backend_job_id_factory.reuse(id);
                }
            })
        } else {
            Box::pin(async {})
        }
    }

    fn get_or_create_persistent_task(
        &self,
        task_type: PersistentTaskType,
        parent_task: TaskId,
        reason: &'static str,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) -> TaskId {
        let result = if let Some(task) = self.task_cache.get(&task_type).map(|task| *task) {
            // fast pass without creating a new task
            self.connect_task_child(parent_task, task, reason, turbo_tasks);

            // TODO maybe force (background) scheduling to avoid inactive tasks hanging in
            // "in progress" until they become active
            task
        } else {
            // slow pass with key lock
            let id = turbo_tasks.get_fresh_task_id();
            let task = match &task_type {
                PersistentTaskType::Native(fn_id, inputs) => {
                    // TODO inputs doesn't need to be cloned when are would be able to get a
                    // reference to the task type stored inside of the task
                    Task::new_native(id, inputs.clone(), *fn_id)
                }
                PersistentTaskType::ResolveNative(fn_id, inputs) => {
                    Task::new_resolve_native(id, inputs.clone(), *fn_id)
                }
                PersistentTaskType::ResolveTrait(trait_type, trait_fn_name, inputs) => {
                    Task::new_resolve_trait(id, *trait_type, trait_fn_name.clone(), inputs.clone())
                }
            };
            // Safety: We have a fresh task id that nobody knows about yet
            unsafe {
                self.memory_tasks.insert(*id, task);
            }
            let (result_task, new) = match self.task_cache.entry(task_type) {
                Entry::Vacant(entry) => {
                    // This is the most likely case
                    entry.insert(id);
                    (id, true)
                }
                Entry::Occupied(entry) => {
                    // Safety: We have a fresh task id that nobody knows about yet
                    unsafe {
                        self.memory_tasks.remove(*id);
                        turbo_tasks.reuse_task_id(id);
                    }
                    (*entry.get(), false)
                }
            };
            self.connect_task_child(parent_task, result_task, reason, turbo_tasks);
            if new {
                self.add_task_to_gc(result_task);
            }
            result_task
        };
        result
    }

    fn create_transient_task(
        &self,
        task_type: TransientTaskType,
        turbo_tasks: &dyn TurboTasksBackendApi,
    ) -> TaskId {
        let id = turbo_tasks.get_fresh_task_id();
        // use INITIAL_SCOPE
        let scope = self.initial_scope;
        self.with_scope(scope, |scope| {
            scope.increment_tasks();
            scope.increment_unfinished_tasks(self);
        });
        let task = match task_type {
            TransientTaskType::Root(f) => Task::new_root(id, scope, move || f() as _),
            TransientTaskType::Once(f) => Task::new_once(id, scope, f),
        };
        // SAFETY: We have a fresh task id where nobody knows about yet
        #[allow(unused_variables)]
        let task = unsafe { self.memory_tasks.insert(*id, task) };
        #[cfg(feature = "print_scope_updates")]
        println!("new {scope} for {task}");
        id
    }
}

pub(crate) enum Job {
    RemoveFromScopes(AutoSet<TaskId>, Vec<TaskScopeId>),
    RemoveFromScope(AutoSet<TaskId>, TaskScopeId),
    ScheduleWhenDirtyFromScope(Vec<TaskId>, &'static str),
    /// Add tasks from a scope. Scheduled by `run_add_from_scope_queue` to
    /// split off work.
    AddToScopeQueue(VecDeque<(TaskId, usize)>, TaskScopeId, bool, &'static str),
    /// Remove tasks from a scope. Scheduled by `run_remove_from_scope_queue` to
    /// split off work.
    RemoveFromScopeQueue(VecDeque<TaskId>, TaskScopeId),
    GarbaggeCollection,
}

impl Job {
    async fn run(self, backend: &MemoryBackend, turbo_tasks: &dyn TurboTasksBackendApi) {
        match self {
            Job::RemoveFromScopes(tasks, scopes) => {
                for task in tasks {
                    backend.with_task(task, |task| {
                        task.remove_from_scopes(scopes.iter().cloned(), backend, turbo_tasks)
                    });
                }
            }
            Job::RemoveFromScope(tasks, scope) => {
                for task in tasks {
                    backend.with_task(task, |task| {
                        task.remove_from_scope(scope, backend, turbo_tasks)
                    });
                }
            }
            Job::ScheduleWhenDirtyFromScope(tasks, reason) => {
                for task in tasks.into_iter() {
                    backend.with_task(task, |task| {
                        task.schedule_when_dirty_from_scope(reason, backend, turbo_tasks);
                    })
                }
            }
            Job::AddToScopeQueue(queue, id, is_optimization_scope, reason) => {
                run_add_to_scope_queue(
                    queue,
                    id,
                    is_optimization_scope,
                    reason,
                    backend,
                    turbo_tasks,
                );
            }
            Job::RemoveFromScopeQueue(queue, id) => {
                run_remove_from_scope_queue(queue, id, backend, turbo_tasks);
            }
            Job::GarbaggeCollection => {
                backend.run_gc(turbo_tasks);
            }
        }
    }
}
