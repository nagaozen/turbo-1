use std::{
    fmt::{Debug, Display},
    time::Duration,
};

/// Stores a Duration in 1/100000th of a second. This allows a precision of 10
/// microseconds for an total duration of 11.9 hours.
/// Values smaller than 10 microseconds are stored as 10 microseconds.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct SmallDuration(u32);

impl SmallDuration {
    pub const ZERO: SmallDuration = SmallDuration(0);
    pub const MIN: SmallDuration = SmallDuration(1);
    pub const MAX: SmallDuration = SmallDuration(u32::MAX);

    pub const fn from_micros(micros: u64) -> Self {
        if micros == 0 {
            return SmallDuration::ZERO;
        }
        if micros <= 10 {
            return SmallDuration::MIN;
        }
        let value = micros / 10;
        if value > u32::MAX as u64 {
            return SmallDuration::MAX;
        }
        SmallDuration(value as u32)
    }

    pub const fn from_millis(millis: u64) -> Self {
        if millis == 0 {
            return SmallDuration::ZERO;
        }
        let value = millis.saturating_mul(100);
        if value > u32::MAX as u64 {
            return SmallDuration::MAX;
        }
        SmallDuration(value as u32)
    }

    pub const fn from_secs(secs: u64) -> Self {
        if secs == 0 {
            return SmallDuration::ZERO;
        }
        let value = secs.saturating_mul(100000);
        if value > u32::MAX as u64 {
            return SmallDuration::MAX;
        }
        SmallDuration(value as u32)
    }
}

impl From<Duration> for SmallDuration {
    fn from(duration: Duration) -> Self {
        if duration.is_zero() {
            return SmallDuration::ZERO;
        }
        let micros = duration.as_micros();
        if micros <= 10 {
            return SmallDuration::MIN;
        }
        (micros / 10)
            .try_into()
            .map_or(SmallDuration::MAX, SmallDuration)
    }
}

impl From<SmallDuration> for Duration {
    fn from(duration: SmallDuration) -> Self {
        Duration::from_micros(duration.0 as u64 * 10)
    }
}

impl Display for SmallDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let duration = Duration::from(*self);
        duration.fmt(f)
    }
}

impl Debug for SmallDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let duration = Duration::from(*self);
        duration.fmt(f)
    }
}
