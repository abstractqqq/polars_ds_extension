


fn isotonic_regression(
    x: &mut [f64],
    w: &mut [f64],
    r: &mut [usize],
) {

    let n = x.len();
    r[0] = 0;
    r[1] = 1;
    let mut b:usize = 0;
    let mut x_pre = x[b];
    let mut w_pre = w[b];

    for i in 1..n {
        b += 1;
        let mut xb = x[i];
        let mut wb = w[i];
        if x_pre > xb {
            b -= 1;
            let mut s = w_pre * x_pre + wb * xb;
            wb += w_pre;
            xb = s / wb;
            let mut j = i;
            while j + 1 < n && xb >= x[j + 1] {
                j += 1;
                s += w[j] * x[j];
                wb += w[j];
                xb = s / wb;
            }
            while b > 0 && x[b-1] >= xb {
                b -= 1;
                s += w[b] * x[b];
                wb += w[b];
                xb = s / wb;
            }
        }
        x_pre = xb;
        x[b] = x_pre;

        w_pre = wb;
        w[b] = w_pre;
        r[b + 1] = i + 1;
    }

    let mut f = n - 1;
    for k in b..0 {
        let t = r[k];
        let xk = x[k];
        for i in f..t {
            x[i] = xk;
        }
        f = t - 1;
    }

}