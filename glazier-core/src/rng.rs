//! xoshiro256++, one stream per site so the CPU and the GPU draw the same way.
//!
//! The generator is written out here rather than pulled from a crate because
//! the CUDA kernel needs the identical arithmetic, and a kernel cannot call
//! into a Rust dependency.

/// One xoshiro256++ stream.
#[derive(Clone, Copy, Debug)]
pub struct Xoshiro {
    s: [u64; 4],
}

impl Xoshiro {
    /// Seed a stream from a 64-bit value through SplitMix64, which is what the
    /// reference implementation recommends for filling the state.
    #[must_use]
    pub fn seed(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// A float in `[0, 1)`, from the top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// A uniform integer in `[0, n)` by rejection, so the result is exact.
    pub fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let v = self.next_u64();
            if v < zone {
                return v % n;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_stay_in_the_unit_interval() {
        let mut r = Xoshiro::seed(7);
        for _ in 0..10_000 {
            let v = r.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn below_covers_its_range_and_stays_inside_it() {
        let mut r = Xoshiro::seed(11);
        let mut seen = [0usize; 5];
        for _ in 0..10_000 {
            let v = r.below(5) as usize;
            seen[v] += 1;
        }
        assert!(seen.iter().all(|&c| c > 1_500), "{seen:?}");
    }

    #[test]
    fn a_seed_reproduces_its_stream() {
        let mut a = Xoshiro::seed(3);
        let mut b = Xoshiro::seed(3);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
