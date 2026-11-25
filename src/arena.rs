//! Double-buffered O(1) ping-pong recurrent state arena helpers.

use core::mem::size_of;

use jero_core::RecurrentStateSlotHeader;

/// Number of `f32` elements occupied by [`RecurrentStateSlotHeader`].
pub const RECURRENT_HEADER_FLOATS: usize = size_of::<RecurrentStateSlotHeader>() / size_of::<f32>();

/// Sentinel stored in `backing[0]` when a ping-pong recurrent header has been initialized.
const RECURRENT_MAGIC_F32: f32 = 1_048_576.0;

/// Manages double-buffered ping-pong recurrent banks (`slot_0`, `slot_1`) inside an arena slice,
/// falling back cleanly to a transient local bank when a compact caller supplies a minimal buffer.
pub(crate) struct PingPongRecurrentArena<'a> {
    backing: &'a mut [f32],
    fallback: Vec<f32>,
    num_slots: usize,
    slot_floats: usize,
    active_bank: usize,
    committed_step: u32,
    uses_arena: bool,
}

impl<'a> PingPongRecurrentArena<'a> {
    pub(crate) fn open(backing: &'a mut [f32], num_slots: usize, slot_floats: usize) -> Self {
        let bank_floats = num_slots.saturating_mul(slot_floats);
        let needed = RECURRENT_HEADER_FLOATS.saturating_add(bank_floats.saturating_mul(2));
        if bank_floats > 0 && backing.len() >= needed {
            let initialized = (backing[0] - RECURRENT_MAGIC_F32).abs() <= f32::EPSILON;
            let active_bank = if initialized {
                usize::from(backing[1] >= 0.5)
            } else {
                0
            };
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "committed_step is stored as an exact non-negative integer in f32"
            )]
            let committed_step = if initialized {
                backing[2].max(0.0) as u32
            } else {
                0
            };
            Self {
                backing,
                fallback: Vec::new(),
                num_slots,
                slot_floats,
                active_bank,
                committed_step,
                uses_arena: true,
            }
        } else {
            Self {
                backing,
                fallback: vec![0.0_f32; bank_floats],
                num_slots,
                slot_floats,
                active_bank: 0,
                committed_step: 0,
                uses_arena: false,
            }
        }
    }

    pub(crate) fn read_slot(&self, slot_idx: usize, dst: &mut [f32]) {
        if slot_idx >= self.num_slots || dst.is_empty() {
            return;
        }
        let bank_stride = self.num_slots * self.slot_floats;
        let off = slot_idx * self.slot_floats;
        let copy_len = dst.len().min(self.slot_floats);
        if self.uses_arena {
            if (self.backing[0] - RECURRENT_MAGIC_F32).abs() > f32::EPSILON {
                dst[..copy_len].fill(0.0);
                return;
            }
            let base = RECURRENT_HEADER_FLOATS + self.active_bank * bank_stride + off;
            dst[..copy_len].copy_from_slice(&self.backing[base..base + copy_len]);
        } else {
            dst[..copy_len].copy_from_slice(&self.fallback[off..off + copy_len]);
        }
    }

    pub(crate) fn write_staging_slot(&mut self, slot_idx: usize, src: &[f32]) {
        if slot_idx >= self.num_slots || src.is_empty() {
            return;
        }
        let bank_stride = self.num_slots * self.slot_floats;
        let off = slot_idx * self.slot_floats;
        let copy_len = src.len().min(self.slot_floats);
        if self.uses_arena {
            let staging_bank = 1 - self.active_bank;
            let base = RECURRENT_HEADER_FLOATS + staging_bank * bank_stride + off;
            self.backing[base..base + copy_len].copy_from_slice(&src[..copy_len]);
        } else {
            self.fallback[off..off + copy_len].copy_from_slice(&src[..copy_len]);
        }
    }

    pub(crate) fn commit(mut self, step: u32) {
        if self.uses_arena {
            let bank_floats = self.num_slots.saturating_mul(self.slot_floats);
            if (self.backing[0] - RECURRENT_MAGIC_F32).abs() > f32::EPSILON {
                self.backing[..RECURRENT_HEADER_FLOATS].fill(0.0);
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "recurrent bank stride fits cleanly in f32 metadata slot"
                )]
                {
                    self.backing[3] = bank_floats as f32;
                }
            }
            let next_bank = 1 - self.active_bank;
            self.backing[0] = RECURRENT_MAGIC_F32;
            self.backing[1] = if next_bank == 1 { 1.0 } else { 0.0 };
            #[expect(
                clippy::cast_precision_loss,
                reason = "inference step count fits cleanly in 24-bit f32 mantissa"
            )]
            {
                self.backing[2] = step as f32;
            }
            self.active_bank = next_bank;
            self.committed_step = step;
        }
    }
}

/// Reads `(active_bank, committed_step)` from a recurrent ping-pong arena slice when initialized.
#[must_use]
pub fn inspect_ping_pong_header(buf: &[f32]) -> Option<(u32, u32)> {
    if buf.len() < RECURRENT_HEADER_FLOATS || (buf[0] - RECURRENT_MAGIC_F32).abs() > f32::EPSILON {
        return None;
    }
    let active_bank = u32::from(buf[1] >= 0.5);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "committed_step encodes an exact non-negative token step index"
    )]
    let committed_step = buf[2].max(0.0) as u32;
    Some((active_bank, committed_step))
}
