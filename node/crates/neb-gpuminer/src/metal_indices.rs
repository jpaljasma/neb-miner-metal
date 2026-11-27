//! ChaCha8 block zero for four independent keys in a word-major layout, so LLVM
//! can execute lane-wise rotations, XORs and wrapping adds on Apple NEON.
#[inline(always)]
fn quarter(x: &mut [[u32; 4]; 16], a: usize, b: usize, c: usize, d: usize) {
    for lane in 0..4 {
        let (mut va, mut vb, mut vc, mut vd) = (x[a][lane], x[b][lane], x[c][lane], x[d][lane]);
        va = va.wrapping_add(vb);
        vd = (vd ^ va).rotate_left(16);
        vc = vc.wrapping_add(vd);
        vb = (vb ^ vc).rotate_left(12);
        va = va.wrapping_add(vb);
        vd = (vd ^ va).rotate_left(8);
        vc = vc.wrapping_add(vd);
        vb = (vb ^ vc).rotate_left(7);
        (x[a][lane], x[b][lane], x[c][lane], x[d][lane]) = (va, vb, vc, vd);
    }
}

pub fn four_keys(keys: &[[u8; 32]; 4], n: u32) -> [[u32; 16]; 4] {
    let mut s = [[0u32; 4]; 16];
    for (i, c) in [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574].into_iter().enumerate() {
        s[i] = [c; 4];
    }
    for (lane, key) in keys.iter().enumerate() {
        for (i, &word) in key.as_chunks::<4>().0.iter().enumerate() {
            s[4 + i][lane] = u32::from_le_bytes(word);
        }
    }
    let mut x = s;
    for _ in 0..4 {
        quarter(&mut x, 0, 4, 8, 12);
        quarter(&mut x, 1, 5, 9, 13);
        quarter(&mut x, 2, 6, 10, 14);
        quarter(&mut x, 3, 7, 11, 15);
        quarter(&mut x, 0, 5, 10, 15);
        quarter(&mut x, 1, 6, 11, 12);
        quarter(&mut x, 2, 7, 8, 13);
        quarter(&mut x, 3, 4, 9, 14);
    }
    std::array::from_fn(|lane| {
        std::array::from_fn(|i| ((u64::from(x[i][lane].wrapping_add(s[i][lane])) * u64::from(n)) >> 32) as u32)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn four_keys_match_reference_indices() {
        for n in [1, 17, 7089, 13404, u32::MAX] {
            for group in 0..64_u32 {
                let keys = std::array::from_fn(|i| *blake3::hash(&(4 * group + i as u32).to_le_bytes()).as_bytes());
                assert_eq!(four_keys(&keys, n), keys.map(|key| neb_pow::shares::share_indices(&key, n)));
            }
        }
    }
}
