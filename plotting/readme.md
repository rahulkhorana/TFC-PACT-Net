
```bash

python3 plotting/stats.py --results_dir=./logs_hyperparameter/qm9 --output_dir=./plotting --main_model=polyatomic --exp_name=qm9
```


```bash
 python plotting/make_latex_table.py  --input ./plotting/<file>.txt --output ./plotting/<file>.tex --test-intervals pm --rename "polyatomic=PACTNet (ECC)" --bold-contains "PACTNet" --val-dec 3 --test-dec 4
```

```bash

python plotting/make_latex_table.py  --input ./plotting/qm9_full_results_summary.txt --output ./plotting/qm9.tex --test-intervals pm --rename "polyatomic=PACTNet (ECC)" --bold-contains "PACTNet" --val-dec 3 --test-dec 4
```