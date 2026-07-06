# Results summary (data/*.npz)

## amplifier.npz
**trajectory amplifier vs simple pooling** (band=all)
- mean  within=0.6716  <- simple
- late  within=0.6970
- max   within=0.6115
- attn  within=0.6845  <- amplifier

## decomp_all.npz
**difficulty vs failure directions**
- cosine(w_fail,w_diff) = 0.2447;  difficulty held-out AUROC = 0.5959
- failure raw = 0.7154;  difficulty-removed = 0.7179

## decomp_deep.npz
**difficulty vs failure directions**
- cosine(w_fail,w_diff) = 0.2404;  difficulty held-out AUROC = 0.5803
- failure raw = 0.7171;  difficulty-removed = 0.7169

## decomp_mid.npz
**difficulty vs failure directions**
- cosine(w_fail,w_diff) = 0.2701;  difficulty held-out AUROC = 0.6474
- failure raw = 0.7174;  difficulty-removed = 0.7218

## decouple.npz
**difficulty/failure decoupling (residualization)** (band=all)
- (1) cos(w_diff,w_fail) = 0.344 (random ~ 0.141)
- (2) within-failure AUROC: base 0.699 -> after removing w_diff 0.659
- (3) difficulty corr^2: base 0.272 -> after removing w_fail 0.213

## ens_all.npz
**ensemble vs best single** (band=all)
- probe    within=0.7126
- pca25    within=0.7332
- spe      within=0.6845
- delta    within=0.6829
- mahal    within=0.6541
- length   within=0.5848
- ensemble z-mean = 0.7387 ; meta = 0.7492 ; best single = 0.7332

## ens_deep.npz
**ensemble vs best single** (band=deep)
- probe    within=0.7122
- pca25    within=0.7328
- spe      within=0.6810
- delta    within=0.6821
- mahal    within=0.6537
- length   within=0.5848
- ensemble z-mean = 0.7324 ; meta = 0.7438 ; best single = 0.7328

## ens_mid.npz
**ensemble vs best single** (band=mid)
- probe    within=0.7363
- pca25    within=0.7438
- spe      within=0.6954
- delta    within=0.7020
- mahal    within=0.6377
- length   within=0.5848
- ensemble z-mean = 0.7531 ; meta = 0.7637 ; best single = 0.7438

## frac_all.npz
**fractional-position emergence (within | cross)**
- frac 0.0-0.2: within=0.5543  cross=0.8329
- frac 0.2-0.4: within=0.5764  cross=0.8242
- frac 0.4-0.6: within=0.6245  cross=0.8440
- frac 0.6-0.8: within=0.6914  cross=0.8635
- frac 0.8-1.0: within=0.6873  cross=0.8956

## frac_deep.npz
**fractional-position emergence (within | cross)**
- frac 0.0-0.2: within=0.5519  cross=0.8317
- frac 0.2-0.4: within=0.5800  cross=0.8262
- frac 0.4-0.6: within=0.6284  cross=0.8449
- frac 0.6-0.8: within=0.6858  cross=0.8614
- frac 0.8-1.0: within=0.6953  cross=0.8959

## frac_mid.npz
**fractional-position emergence (within | cross)**
- frac 0.0-0.2: within=0.5543  cross=0.8292
- frac 0.2-0.4: within=0.5582  cross=0.8047
- frac 0.4-0.6: within=0.5962  cross=0.8180
- frac 0.6-0.8: within=0.6970  cross=0.8596
- frac 0.8-1.0: within=0.6860  cross=0.8974

## mc_all.npz
**manifold-constraint SPE: within-AUROC vs healthy-subspace dim k** (band=all, agg=latemean)
- k=1    within=0.6468  cross=0.7051
- k=2    within=0.6581  cross=0.7158
- k=5    within=0.6645  cross=0.6808
- k=10   within=0.6771  cross=0.6925
- k=25   within=0.6827  cross=0.7076
- k=50   within=0.6794  cross=0.7012
- k=100  within=0.6788  cross=0.6994
- k=200  within=0.6867  cross=0.6921
- ||z|| norm baseline (within) = 0.5694

## norm_all.npz
**normalization / metric variants** (band=?)
- PR raw: 0.5460
- PR zscore(diag): 0.5285
- PR drop-top50: 0.5452
- PR PCA-whiten: 0.5125
- Mahal diag: 0.6057
- Mahal PCA100: 0.6037
- Mahal++ (L2norm): 0.6170

## norm_deep.npz
**normalization / metric variants** (band=?)
- PR raw: 0.5550
- PR zscore(diag): 0.5242
- PR drop-top50: 0.5289
- PR PCA-whiten: 0.5098
- Mahal diag: 0.6073
- Mahal PCA100: 0.6080
- Mahal++ (L2norm): 0.6170

## norm_mid.npz
**normalization / metric variants** (band=?)
- PR raw: 0.5917
- PR zscore(diag): 0.5573
- PR drop-top50: 0.5238
- PR PCA-whiten: 0.5390
- Mahal diag: 0.5823
- Mahal PCA100: 0.6108
- Mahal++ (L2norm): 0.6061

## pairwise.npz
**pooled vs within-problem PAIRWISE training** (PCA=25)
- mid    pooled=0.7429  pairwise=0.7337  delta=-0.0092

## probe_all.npz
**learned probe + difficulty inflation**
- within-problem probe (HONEST) = 0.7149 +/- 0.0078
- GROUP-kfold pooled (held-out probs, pooled metric) = 0.7152  <- same held-out preds as within, pooled
- cross-problem pooled (random split, INFLATED) = 0.8984
- length baseline = 0.5848   unsupervised = 0.5261

## probe_deep.npz
**learned probe + difficulty inflation**
- within-problem probe (HONEST) = 0.7116 +/- 0.0074
- GROUP-kfold pooled (held-out probs, pooled metric) = 0.7117  <- same held-out preds as within, pooled
- cross-problem pooled (random split, INFLATED) = 0.8973
- length baseline = 0.5848   unsupervised = 0.5328

## probe_interpret.npz
**probe weight w interpretation** (band=last)
- cos(w, mean-diff incorrect-correct) = 0.158; cos(w, mean-act) = -0.014; cos(w, sigma) = -0.010
- sparsity: eff #neurons (PR) = 1127.5; 50% mass in 915 neurons, 90% in 2588
- INTERPRETABLE dir (mean-diff) detector: within=0.672 cross=0.748
- logit-lens CENTERED TOP: ['efon', 'veren', 'raki', 'PKG', 'نم', 'це', 'jur', 'rains', 'iali', '436', 'ondo', '.xhtml', '賀', '统', 'onas']
- logit-lens CENTERED BOT: ['立て', 'olle', 'lickr', '��️', 'destinations', 'Disappear', 'ulu', '.microsoft', 'escape', 'nesty', 'barang', 'SSION', 'Slut', 'igy', 'utterstock']
- logit-lens RAW TOP (artifact): ['efon', 'veren', 'raki', 'PKG', 'نم', 'це', 'jur', 'rains', 'iali', '436']

## probe_mid.npz
**learned probe + difficulty inflation**
- within-problem probe (HONEST) = 0.7069 +/- 0.0061
- GROUP-kfold pooled (held-out probs, pooled metric) = 0.7299  <- same held-out preds as within, pooled
- cross-problem pooled (random split, INFLATED) = 0.8958
- length baseline = 0.5848   unsupervised = 0.5554

## sparse_all.npz
**sparse(L1) vs low-rank(PCA)**
- L1 (C: AUROC, #nonzero): 0.0005:0.5000/0, 0.001:0.5000/0, 0.003:0.6739/1, 0.01:0.7243/35, 0.03:0.7192/129, 0.1:0.7038/317
- PCA (k: AUROC): 2:0.5698, 5:0.6529, 10:0.7098, 25:0.7275, 50:0.7178, 100:0.7023, 300:0.6876

## sparse_deep.npz
**sparse(L1) vs low-rank(PCA)**
- L1 (C: AUROC, #nonzero): 0.0005:0.5000/0, 0.001:0.5000/0, 0.003:0.6712/1, 0.01:0.7154/33, 0.03:0.7243/130, 0.1:0.7085/313
- PCA (k: AUROC): 2:0.5618, 5:0.6594, 10:0.7168, 25:0.7352, 50:0.7166, 100:0.7077, 300:0.6867

## sparse_mid.npz
**sparse(L1) vs low-rank(PCA)**
- L1 (C: AUROC, #nonzero): 0.0005:0.5000/0, 0.001:0.5000/0, 0.003:0.6806/1, 0.01:0.7232/34, 0.03:0.7305/137, 0.1:0.7185/340
- PCA (k: AUROC): 2:0.5872, 5:0.6469, 10:0.7187, 25:0.7205, 50:0.7174, 100:0.6998, 300:0.6668

## temporal_all.npz
**per-step-position (within | cross)**
- pos 0: within=0.5117  cross=0.7872
- pos 1: within=0.5683  cross=0.7946
- pos 2: within=0.5300  cross=0.7780
- pos 3: within=0.5530  cross=0.7799
- pos 4: within=0.5612  cross=0.7807
- pos 5: within=0.6337  cross=0.8008
- pos 6: within=0.6107  cross=0.7956
- pos 7: within=0.5872  cross=0.7771
- pos 8: within=0.6139  cross=0.7790
- pos 9: within=0.6231  cross=0.7608
- pos 10: within=0.5986  cross=0.7669
- pos 11: within=0.6113  cross=0.7756
  windows: early[0:3]=0.5446, mid[2:6]=0.6018, late[-3:]=0.7097, all-mean=0.7113

## temporal_deep.npz
**per-step-position (within | cross)**
- pos 0: within=0.5109  cross=0.7877
- pos 1: within=0.5677  cross=0.7965
- pos 2: within=0.5363  cross=0.7804
- pos 3: within=0.5484  cross=0.7830
- pos 4: within=0.5673  cross=0.7852
- pos 5: within=0.6406  cross=0.8042
- pos 6: within=0.6048  cross=0.7958
- pos 7: within=0.5859  cross=0.7771
- pos 8: within=0.6182  cross=0.7790
- pos 9: within=0.6101  cross=0.7588
- pos 10: within=0.5841  cross=0.7660
- pos 11: within=0.6107  cross=0.7744
  windows: early[0:3]=0.5468, mid[2:6]=0.6040, late[-3:]=0.7136, all-mean=0.7081

## temporal_mid.npz
**per-step-position (within | cross)**
- pos 0: within=0.5126  cross=0.7772
- pos 1: within=0.5436  cross=0.7658
- pos 2: within=0.5103  cross=0.7395
- pos 3: within=0.5479  cross=0.7522
- pos 4: within=0.5534  cross=0.7450
- pos 5: within=0.6055  cross=0.7656
- pos 6: within=0.5807  cross=0.7575
- pos 7: within=0.5922  cross=0.7486
- pos 8: within=0.6161  cross=0.7627
- pos 9: within=0.6171  cross=0.7513
- pos 10: within=0.6171  cross=0.7642
- pos 11: within=0.6203  cross=0.7529
  windows: early[0:3]=0.5263, mid[2:6]=0.5813, late[-3:]=0.7127, all-mean=0.7079
