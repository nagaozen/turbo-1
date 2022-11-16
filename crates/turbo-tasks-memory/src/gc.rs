use std::{
    fmt::{Display, Formatter},
    time::Duration,
};

use turbo_tasks::{CellId, TaskId};

#[derive(Debug)]
pub struct GcTaskInfo {
    pub unread_cells: usize,
    pub cells: Vec<(CellId, Vec<TaskId>)>,
    pub output: Vec<TaskId>,
    pub compute_duration: Duration,
    pub age: Duration,
    pub active: bool,
}

#[derive(Debug)]
pub enum GcAction {
    UnreadCells(TaskId),
    ReadCell(TaskId, CellId),
    Unload(TaskId),
}

#[derive(Debug)]
pub struct GcItem {
    pub action: GcAction,
    pub compute_duration: Duration,
    pub age: Duration,
}

impl Display for GcAction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GcAction::UnreadCells(task_id) => {
                write!(f, "Unread cells for {}", task_id)
            }
            GcAction::ReadCell(task_id, cell_id) => {
                write!(f, "Read cell {} for {}", cell_id, task_id)
            }
            GcAction::Unload(task_id) => {
                write!(f, "Unload {}", task_id)
            }
        }
    }
}

impl GcItem {
    pub fn cmp_priority(&self, other: &Self) -> std::cmp::Ordering {
        if !other.compute_duration.is_zero() && other.compute_duration > self.compute_duration * 2 {
            return std::cmp::Ordering::Less;
        } else if !self.compute_duration.is_zero()
            && self.compute_duration > other.compute_duration * 2
        {
            return std::cmp::Ordering::Greater;
        }
        if !other.age.is_zero() && other.age > self.age * 2 {
            return std::cmp::Ordering::Less;
        } else if !self.age.is_zero() && self.age > other.age * 2 {
            return std::cmp::Ordering::Greater;
        }
        std::cmp::Ordering::Equal
    }

    pub fn task(&self) -> TaskId {
        match self.action {
            GcAction::UnreadCells(task_id) => task_id,
            GcAction::ReadCell(task_id, _) => task_id,
            GcAction::Unload(task_id) => task_id,
        }
    }

    pub fn cmp_task(&self, other: &Self) -> std::cmp::Ordering {
        self.task().cmp(&other.task())
    }
}
