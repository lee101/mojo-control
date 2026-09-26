"""Dense kernels for state-space and transfer-function evaluation."""

from std.sys import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


@always_inline
def tf_eval_range(
    num: FPtr,
    den: FPtr,
    nnum: Int,
    nden: Int,
    xr: FPtr,
    xi: FPtr,
    real: FPtr,
    imag: FPtr,
    start: Int,
    end: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var k = start
    while k + W <= end:
        var xrv = xr.load[width=W](k)
        var xiv = xi.load[width=W](k)
        var nr = SIMD[DType.float64, W](num[0])
        var ni = SIMD[DType.float64, W](0.0)
        for j in range(1, nnum):
            var old_nr = nr
            nr = old_nr * xrv - ni * xiv + num[j]
            ni = old_nr * xiv + ni * xrv
        var dr = SIMD[DType.float64, W](den[0])
        var di = SIMD[DType.float64, W](0.0)
        for j in range(1, nden):
            var old_dr = dr
            dr = old_dr * xrv - di * xiv + den[j]
            di = old_dr * xiv + di * xrv
        var scale = dr * dr + di * di
        real.store(k, (nr * dr + ni * di) / scale)
        imag.store(k, (ni * dr - nr * di) / scale)
        k += W

    while k < end:
        var nr = num[0]
        var ni = 0.0
        for j in range(1, nnum):
            var old_nr = nr
            nr = old_nr * xr[k] - ni * xi[k] + num[j]
            ni = old_nr * xi[k] + ni * xr[k]
        var dr = den[0]
        var di = 0.0
        for j in range(1, nden):
            var old_dr = dr
            dr = old_dr * xr[k] - di * xi[k] + den[j]
            di = old_dr * xi[k] + di * xr[k]
        var scale = dr * dr + di * di
        real[k] = (nr * dr + ni * di) / scale
        imag[k] = (ni * dr - nr * di) / scale
        k += 1


@export("mctl_tf_eval")
def mctl_tf_eval(
    num_addr: Int,
    den_addr: Int,
    nnum: Int,
    nden: Int,
    xr_addr: Int,
    xi_addr: Int,
    real_addr: Int,
    imag_addr: Int,
    count: Int,
) abi("C"):
    var num = fp(num_addr)
    var den = fp(den_addr)
    var xr = fp(xr_addr)
    var xi = fp(xi_addr)
    var real = fp(real_addr)
    var imag = fp(imag_addr)
    comptime PARALLEL_THRESHOLD = 131072
    comptime CHUNK_SIZE = 8192
    if count < PARALLEL_THRESHOLD:
        tf_eval_range(num, den, nnum, nden, xr, xi, real, imag, 0, count)
    else:
        var chunks = (count + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk in range(chunks):
            var start = chunk * CHUNK_SIZE
            var end = min(start + CHUNK_SIZE, count)
            tf_eval_range(num, den, nnum, nden, xr, xi, real, imag, start, end)


@export("mctl_ss_eval")
def mctl_ss_eval(
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    d_addr: Int,
    n: Int,
    m: Int,
    p: Int,
    xr_addr: Int,
    xi_addr: Int,
    real_addr: Int,
    imag_addr: Int,
    mat_r_addr: Int,
    mat_i_addr: Int,
    rhs_r_addr: Int,
    rhs_i_addr: Int,
    status_addr: Int,
    count: Int,
) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var c = fp(c_addr)
    var d = fp(d_addr)
    var xr = fp(xr_addr)
    var xi = fp(xi_addr)
    var real = fp(real_addr)
    var imag = fp(imag_addr)
    var mat_r = fp(mat_r_addr)
    var mat_i = fp(mat_i_addr)
    var rhs_r = fp(rhs_r_addr)
    var rhs_i = fp(rhs_i_addr)
    var status = ip(status_addr)

    for f in range(count):
        status[f] = 0
        for i in range(n):
            for j in range(n):
                var idx = i * n + j
                mat_r[idx] = -a[idx]
                mat_i[idx] = 0.0
                if i == j:
                    mat_r[idx] += xr[f]
                    mat_i[idx] = xi[f]
            for q in range(m):
                rhs_r[i * m + q] = b[i * m + q]
                rhs_i[i * m + q] = 0.0

        for k in range(n):
            var pivot = k
            var pivot_norm = (
                mat_r[k * n + k] * mat_r[k * n + k]
                + mat_i[k * n + k] * mat_i[k * n + k]
            )
            for i in range(k + 1, n):
                var candidate = (
                    mat_r[i * n + k] * mat_r[i * n + k]
                    + mat_i[i * n + k] * mat_i[i * n + k]
                )
                if candidate > pivot_norm:
                    pivot = i
                    pivot_norm = candidate
            if pivot_norm <= 1.0e-30:
                status[f] = 1
                break
            if pivot != k:
                for j in range(k, n):
                    var kj = k * n + j
                    var pj = pivot * n + j
                    var tr = mat_r[kj]
                    var ti = mat_i[kj]
                    mat_r[kj] = mat_r[pj]
                    mat_i[kj] = mat_i[pj]
                    mat_r[pj] = tr
                    mat_i[pj] = ti
                for q in range(m):
                    var kq = k * m + q
                    var pq = pivot * m + q
                    var tr = rhs_r[kq]
                    var ti = rhs_i[kq]
                    rhs_r[kq] = rhs_r[pq]
                    rhs_i[kq] = rhs_i[pq]
                    rhs_r[pq] = tr
                    rhs_i[pq] = ti

            var pr = mat_r[k * n + k]
            var pi = mat_i[k * n + k]
            var pscale = pr * pr + pi * pi
            for i in range(k + 1, n):
                var ik = i * n + k
                var fr = (mat_r[ik] * pr + mat_i[ik] * pi) / pscale
                var fi = (mat_i[ik] * pr - mat_r[ik] * pi) / pscale
                for j in range(k + 1, n):
                    var ij = i * n + j
                    var kj = k * n + j
                    var ur = mat_r[kj]
                    var ui = mat_i[kj]
                    mat_r[ij] -= fr * ur - fi * ui
                    mat_i[ij] -= fr * ui + fi * ur
                for q in range(m):
                    var iq = i * m + q
                    var kq = k * m + q
                    var rr = rhs_r[kq]
                    var ri = rhs_i[kq]
                    rhs_r[iq] -= fr * rr - fi * ri
                    rhs_i[iq] -= fr * ri + fi * rr

        if status[f] == 0:
            for rev in range(n):
                var i = n - 1 - rev
                var ur = mat_r[i * n + i]
                var ui = mat_i[i * n + i]
                var uscale = ur * ur + ui * ui
                for q in range(m):
                    var rr = rhs_r[i * m + q]
                    var ri = rhs_i[i * m + q]
                    for j in range(i + 1, n):
                        var mr = mat_r[i * n + j]
                        var mi = mat_i[i * n + j]
                        var sr = rhs_r[j * m + q]
                        var si = rhs_i[j * m + q]
                        rr -= mr * sr - mi * si
                        ri -= mr * si + mi * sr
                    rhs_r[i * m + q] = (rr * ur + ri * ui) / uscale
                    rhs_i[i * m + q] = (ri * ur - rr * ui) / uscale

        for row in range(p):
            for q in range(m):
                var vr = d[row * m + q]
                var vi = 0.0
                if status[f] == 0:
                    for i in range(n):
                        vr += c[row * n + i] * rhs_r[i * m + q]
                        vi += c[row * n + i] * rhs_i[i * m + q]
                var dst = (f * p + row) * m + q
                real[dst] = vr
                imag[dst] = vi


@export("mctl_ss_simulate")
def mctl_ss_simulate(
    a_addr: Int,
    b0_addr: Int,
    b1_addr: Int,
    c_addr: Int,
    d_addr: Int,
    u_addr: Int,
    x_addr: Int,
    next_addr: Int,
    y_addr: Int,
    states_addr: Int,
    steps: Int,
    n: Int,
    m: Int,
    p: Int,
) abi("C"):
    var a = fp(a_addr)
    var b0 = fp(b0_addr)
    var b1 = fp(b1_addr)
    var c = fp(c_addr)
    var d = fp(d_addr)
    var u = fp(u_addr)
    var x = fp(x_addr)
    var next_x = fp(next_addr)
    var y = fp(y_addr)
    var states = fp(states_addr)

    for k in range(steps):
        for i in range(n):
            states[k * n + i] = x[i]
        for row in range(p):
            var value = 0.0
            for i in range(n):
                value += c[row * n + i] * x[i]
            for q in range(m):
                value += d[row * m + q] * u[k * m + q]
            y[k * p + row] = value
        if k + 1 < steps:
            for i in range(n):
                var value = 0.0
                for j in range(n):
                    value += a[i * n + j] * x[j]
                for q in range(m):
                    value += b0[i * m + q] * u[k * m + q]
                    value += b1[i * m + q] * u[(k + 1) * m + q]
                next_x[i] = value
            for i in range(n):
                x[i] = next_x[i]
