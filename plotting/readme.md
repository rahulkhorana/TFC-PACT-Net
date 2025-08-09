```bash
 python plotting/make_latex_table.py  --input ./plotting/<file>.txt --output ./plotting/<file>.tex --test-intervals pm --rename "polyatomic=PACTNet (ECC)" --bold-contains "PACTNet" --val-dec 3 --test-dec 4
```

```bash
python3 plotting/stats.py \
  --results_dir ./logs_hyperparameter/qm9 \
  --output_dir ./plotting \
  --exp_name qm9 \
  --control_model polyatomic_polyatomic \
  --k 5 \
  --alpha 0.05 \
  --print_ci


python3 plotting/stats.py \
  --results_dir ./logs_hyperparameter/lipophil \
  --output_dir ./plotting \
  --exp_name lipophil \
  --control_model polyatomic_polyatomic \
  --k 5 \
  --alpha 0.05 \
  --print_ci



python3 plotting/stats.py \
  --results_dir ./logs_hyperparameter/freesolv \
  --output_dir ./plotting \
  --exp_name freesolv \
  --control_model polyatomic_polyatomic \
  --k 5 \
  --alpha 0.05 \
  --print_ci


python3 plotting/stats.py \
  --results_dir ./logs_hyperparameter/esol \
  --output_dir ./plotting \
  --exp_name esol \
  --control_model polyatomic_polyatomic \
  --k 5 \
  --alpha 0.05 \
  --print_ci



python3 plotting/stats.py \
  --results_dir ./logs_hyperparameter/boilingpoint \
  --output_dir ./plotting \
  --exp_name boilingpoint \
  --control_model polyatomic_polyatomic \
  --k 5 \
  --alpha 0.05 \
  --print_ci



python3 plotting/stats.py \
  --results_dir ./logs_hyperparameter/bindingdb \
  --output_dir ./plotting \
  --exp_name bindingdb \
  --control_model polyatomic_polyatomic \
  --k 5 \
  --alpha 0.05 \
  --print_ci


python3 plotting/stats.py \
  --results_dir ./logs_hyperparameter/ic50 \
  --output_dir ./plotting \
  --exp_name ic50 \
  --control_model polyatomic_polyatomic \
  --k 5 \
  --alpha 0.05 \
  --print_ci
```