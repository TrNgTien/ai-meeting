//! Enumerating microphones for the device dropdown.
//!
//! Port of `audio_capture.InputDevice` / `list_input_devices` / `default_input_device`.
//!
//! One deliberate improvement: the Python app identified devices by PortAudio's
//! integer index, which is only stable within a single process run — unplug a
//! headset and every later index shifts. cpal exposes a stable string `DeviceId`,
//! so the selection survives a relaunch and a changed device list.

use cpal::traits::{DeviceTrait, HostTrait};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputDevice {
    /// Stable across launches, unlike the Python app's PortAudio index.
    pub id: String,
    pub name: String,
    pub channels: u16,
    pub sample_rate: u32,
    pub is_default: bool,
}

/// Every input device that reports a usable default configuration.
///
/// A device that fails to describe itself is skipped rather than surfaced as a
/// broken entry: a dropdown row that cannot be recorded from is worse than one
/// that isn't offered.
pub fn list_input_devices() -> Vec<InputDevice> {
    let host = cpal::default_host();
    let default_id = host
        .default_input_device()
        .and_then(|device| device.id().ok())
        .map(|id| id.to_string());

    let Ok(devices) = host.input_devices() else {
        return Vec::new();
    };

    devices
        .filter_map(|device| describe(&device, default_id.as_deref()))
        .collect()
}

pub fn default_input_device() -> Option<InputDevice> {
    let host = cpal::default_host();
    let device = host.default_input_device()?;
    let id = device.id().ok().map(|id| id.to_string());
    describe(&device, id.as_deref())
}

fn describe(device: &cpal::Device, default_id: Option<&str>) -> Option<InputDevice> {
    let id = device.id().ok()?.to_string();
    let name = device
        .description()
        .ok()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|| id.clone());
    let config = device.default_input_config().ok()?;

    Some(InputDevice {
        is_default: default_id == Some(id.as_str()),
        id,
        name,
        channels: config.channels(),
        sample_rate: config.sample_rate(),
    })
}

/// Find a device by the id the UI sent back, falling back to the system default.
///
/// The fallback matters: a user who recorded with a headset last week and opens
/// the app without it should get a working recording, not an error.
pub fn resolve_input_device(id: Option<&str>) -> Option<cpal::Device> {
    let host = cpal::default_host();
    if let Some(wanted) = id {
        if let Ok(devices) = host.input_devices() {
            for device in devices {
                if device.id().ok().map(|found| found.to_string()).as_deref() == Some(wanted) {
                    return Some(device);
                }
            }
        }
    }
    host.default_input_device()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumeration_does_not_panic_on_this_machine() {
        // Cannot assert on hardware in CI, but the code path must be safe to run
        // with any device list, including an empty one.
        let devices = list_input_devices();
        for device in &devices {
            assert!(!device.id.is_empty());
            assert!(!device.name.is_empty(), "a nameless row is not selectable");
            assert!(device.sample_rate > 0);
        }
        // At most one device may claim to be the default.
        assert!(devices.iter().filter(|d| d.is_default).count() <= 1);
    }

    #[test]
    fn resolving_an_unknown_id_falls_back_to_the_default() {
        // Whatever this machine has, an unknown id must not produce a device
        // that claims to be the one asked for.
        let resolved = resolve_input_device(Some("no-such-device-id"));
        let default = default_input_device();
        match (resolved, default) {
            (Some(device), Some(expected)) => {
                let id = device.id().ok().map(|id| id.to_string());
                assert_eq!(id.as_deref(), Some(expected.id.as_str()));
            }
            // A machine with no input at all is a valid state.
            (None, None) => {}
            (resolved, expected) => panic!(
                "inconsistent: resolved={:?}, default={:?}",
                resolved.map(|d| d.id().ok()),
                expected
            ),
        }
    }
}
