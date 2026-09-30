//! Qwen3.5's Gated DeltaNet recurrence on the GPU: mlx-lm's Metal kernel
//! (`models/gated_delta.py`, scalar gating, no mask), called through MLX's C
//! API because mlx-rs 0.32 doesn't wrap `mx.fast.metal_kernel`. One thread
//! group of 32 × 4 per (value head, 4 value dims); each SIMD group walks the
//! tokens in order, carrying its slice of the `[Dv, Dk]` state in registers.
//! An ops version of the same step (`reference`) checks it (`bun run test:llm`).

use super::model::{e, R};
use mlx_rs::{Array, Stream};
use std::ffi::CString;

const SOURCE: &str = r#"
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
"#;

fn check(code: i32, what: &str) -> R<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("gated delta kernel: {what} failed"))
    }
}

/// Runs `T` recurrent steps. Shapes: `q`, `k` `[B, T, Hk, Dk]`; `v`
/// `[B, T, Hv, Dv]`; `g`, `beta` `[B, T, Hv]`; `state` `[B, Hv, Dv, Dk]`
/// (float32). Returns `y` `[B, T, Hv, Dv]` in `q`'s dtype and the new state.
pub fn update(q: &Array, k: &Array, v: &Array, g: &Array, beta: &Array, state: &Array) -> R<(Array, Array)> {
    let (qs, vs) = (q.shape(), v.shape());
    let (b, t, hk, dk) = (qs[0], qs[1], qs[2], qs[3]);
    let (hv, dv) = (vs[2], vs[3]);
    let steps = Array::from_int(t);
    let in_t: mlx_sys::mlx_dtype = q.dtype().into();
    let st_t: mlx_sys::mlx_dtype = state.dtype().into();
    let name = CString::new("gated_delta_step").map_err(e)?;
    let source = CString::new(SOURCE).map_err(e)?;
    let empty = CString::new("").map_err(e)?;
    let names = |list: &[&str]| -> R<mlx_sys::mlx_vector_string> {
        let v = unsafe { mlx_sys::mlx_vector_string_new() };
        for s in list {
            let c = CString::new(*s).map_err(e)?;
            check(unsafe { mlx_sys::mlx_vector_string_append_value(v, c.as_ptr()) }, "names")?;
        }
        Ok(v)
    };
    let input_names = names(&["q", "k", "v", "g", "beta", "state_in", "T"])?;
    let output_names = names(&["y", "state_out"])?;
    let stream = Stream::thread_local_or_default();
    // SAFETY: every handle made here is freed below; the inputs are borrowed for the
    // call (the vector holds its own references), and the outputs are moved into
    // `Array`s, which free them.
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(name.as_ptr(), input_names, output_names, source.as_ptr(), empty.as_ptr(), true, false);
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        let inputs_raw = [q.as_ptr(), k.as_ptr(), v.as_ptr(), g.as_ptr(), beta.as_ptr(), state.as_ptr(), steps.as_ptr()];
        let inputs = mlx_sys::mlx_vector_array_new_data(inputs_raw.as_ptr(), inputs_raw.len());
        let mut outputs = mlx_sys::mlx_vector_array_new();
        let result = (|| -> R<(Array, Array)> {
            for (tname, dt) in [("InT", in_t), ("StT", st_t)] {
                let c = CString::new(tname).map_err(e)?;
                check(mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(config, c.as_ptr(), dt), "dtype arg")?;
            }
            for (tname, val) in [("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv)] {
                let c = CString::new(tname).map_err(e)?;
                check(mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c.as_ptr(), val), "int arg")?;
            }
            check(mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 32, dv, b * hv), "grid")?;
            check(mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1), "thread group")?;
            let y_shape = [b, t, hv, dv];
            check(mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(config, y_shape.as_ptr(), 4, in_t), "output y")?;
            let s_shape = state.shape();
            check(mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(config, s_shape.as_ptr(), s_shape.len(), st_t), "output state")?;
            check(mlx_sys::mlx_fast_metal_kernel_apply(&mut outputs, kernel, inputs, config, stream.as_ptr()), "apply")?;
            let mut y = mlx_sys::mlx_array_new();
            let mut s = mlx_sys::mlx_array_new();
            let (ry, rs) = (mlx_sys::mlx_vector_array_get(&mut y, outputs, 0), mlx_sys::mlx_vector_array_get(&mut s, outputs, 1));
            let (y, s) = (Array::from_ptr(y), Array::from_ptr(s));
            check(ry, "read y")?;
            check(rs, "read state")?;
            Ok((y, s))
        })();
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_fast_metal_kernel_free(kernel);
        mlx_sys::mlx_vector_string_free(input_names);
        mlx_sys::mlx_vector_string_free(output_names);
        result
    }
}

/// The same recurrence with MLX ops, one step at a time (mlx-lm
/// `gated_delta_ops`): the check for the kernel.
#[cfg(test)]
pub fn reference(q: &Array, k: &Array, v: &Array, g: &Array, beta: &Array, state: &Array) -> R<(Array, Array)> {
    use mlx_rs::ops;
    use mlx_rs::ops::indexing::IndexOp;
    let (hk, hv) = (q.shape()[2], v.shape()[2]);
    let (q, k) = if hv > hk { (ops::repeat_axis::<f32>(q.clone(), hv / hk, -2).map_err(e)?, ops::repeat_axis::<f32>(k.clone(), hv / hk, -2).map_err(e)?) } else { (q.clone(), k.clone()) };
    let mut s = state.clone();
    let mut ys = Vec::new();
    for t in 0..q.shape()[1] {
        let (qt, kt, vt) = (q.index((.., t)), k.index((.., t)), v.index((.., t)));
        let (gt, bt) = (g.index((.., t)), beta.index((.., t)));
        s = ops::multiply(&s, gt.expand_dims_axes(&[-1, -2]).map_err(e)?).map_err(e)?;
        let kx = kt.expand_dims(-2).map_err(e)?;
        let kv_mem = ops::multiply(&s, &kx).and_then(|x| x.sum_axis(-1, false)).map_err(e)?;
        let delta = ops::multiply(ops::subtract(&vt, &kv_mem).map_err(e)?, bt.expand_dims(-1).map_err(e)?).map_err(e)?;
        s = ops::add(&s, ops::multiply(&kx, delta.expand_dims(-1).map_err(e)?).map_err(e)?).map_err(e)?;
        let y = ops::multiply(&s, qt.expand_dims(-2).map_err(e)?).and_then(|x| x.sum_axis(-1, false)).map_err(e)?;
        ys.push(y.as_dtype(q.dtype()).map_err(e)?);
    }
    Ok((ops::stack(&ys, 1).map_err(e)?, s))
}
