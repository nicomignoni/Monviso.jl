# API

The convention used for naming indexed terms is the following:

- Indexed terms' names are single letters
- `xk` stands for $x_k$
- `xkn` stands for $x_{k+n}$ 
- `xnk` stands for $x_{k-n}$

Therefore, as examples, `y0` is $y_0$, `tk` is $t_k$, `z1k` is $z_{k-1}$, and `sk2` is $s_{k+2}$.
The vector corresponding to the decision variable of the VI is always denoted with $\mathbf{x}$; all other vectors that might be used or returned are generically referred to as *auxiliary points*.

```@docs
prox
```

```@docs
pg
```

```@docs
eg
```

```@docs
popov
```

```@docs
fbf
```

```@docs
frb
```

```@docs
prg 
```

```@docs
eag
```

```@docs
arg
```

```@docs
fogda
```

```@docs
cfogda
```

```@docs
graal
```

```@docs
agraal
```

```@docs
hgraal_1
```
