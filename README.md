# Fiber ODF estimation with a deep neural network

This code contains the code for the papers:

- [Robust Estimation of the Microstructure of the Early Developing Brain Using Deep Learning](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_28) by Hamza Kebiri, Ali Gholipour, Rizhong Lin, Lana Vasung and Davood Karimi\*, Meritxell Bach Cuadra\* (\*: Equal contribution), and
- [Deep learning microstructure estimation of developing brains from diffusion MRI: a newborn and fetal study](https://www.biorxiv.org/content/10.1101/2023.07.01.547351v1) by Hamza Kebiri, Ali Gholipour, Rizhong Lin, Lana Vasung, Camilo Calixto, Željka Krsnik and Davood Karimi\*, Meritxell Bach Cuadra\* (\*: Equal contribution)

## Structure

- `fod_cnn_genData.py` to generate the input data, ground truth data and CSD baseline from dHCP data
- `fod_cnn_train.py` to train the CNN to learn FOD predictions from input data
- `fod_cnn_test.py` to test the CNN
- `fod_cnn_genDataGS.py` to generate the two gold standards (GS1 and GS2)
- `dk_aux.py`, `dk_model.py`, `crl_aux.py` and `dk_seg.py` include auxiliary functions (developed by [Davood Karimi](mailto:davood.karimi@childrens.harvard.edu)) that are used by the scripts above.
- `MLP` folder contains TrainMLP.py to train the MLP model of Karimi et al., Neuroimage, 2021
- `CTtrack` folder contains the code of Hosseini et al., Neuroscience Informatics, 2022

Please note that the network code has been reimplemented by [Rizhong Lin](mailto:rizhong.lin@epfl.ch) with a newer version of TensorFlow [here](https://github.com/Medical-Image-Analysis-Laboratory/dl_fiber_domain_shift/tree/main/DeepLearning/kebiri_robust_2023). 

## Data

The data used in this study are from the [Developing Human Connectome Project (dHCP)](https://www.humanconnectome.org/study/lifespan-developing-human-connectome-project), and a private dataset of fetal MRI scans which is not publicly available.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{kebiri_deep_2024,
  title   = {Deep learning microstructure estimation of developing brains from diffusion MRI: A newborn and fetal study},
  doi     = {10.1101/2023.07.01.547351},
  author  = {Kebiri, Hamza and Gholipour, Ali and Lin, Rizhong and Vasung, Lana and Calixto, Camilo and Krsnik, Željka and Karimi, Davood and Bach Cuadra, Meritxell},
  year    = {2024},
  journal = {Medical Image Analysis},
}

@inproceedings{kebiri_robust_2023,
  title     = {Robust {Estimation} of the {Microstructure} of the {Early} {Developing} {Brain} {Using} {Deep} {Learning}},
  url       = {http://link.springer.com/chapter/10.1007/978-3-031-43990-2_28},
  doi       = {10.1007/978-3-031-43990-2_28},
  author    = {Kebiri, Hamza and Gholipour, Ali and Lin, Rizhong and Vasung, Lana and Karimi, Davood and Bach Cuadra, Meritxell},
  year      = 2023,
  month     = oct,
  booktitle = {26th {International} {Conference} on {Medical} {Image} {Computing} and {Computer} {Assisted} {Intervention} -- {MICCAI} 2023},
  pages     = {293--303}
}

@inproceedings{lin_cross-age_2024,
  title     = {Cross-{Age} and {Cross}-{Site} {Domain} {Shift} {Impacts} on {Deep} {Learning}-{Based} {White} {Matter} {Fiber} {Estimation} in {Newborn} and {Baby} {Brains}},
  doi       = {10.48550/arXiv.2312.14773},
  author    = {Lin, Rizhong and Gholipour, Ali and Thiran, Jean-Philippe and Karimi, Davood and Kebiri, Hamza and Bach Cuadra, Meritxell},
  year      = 2024,
  month     = may,
  booktitle = {21st {IEEE} {International} {Symposium} on {Biomedical} {Imaging} ({ISBI})}
}
```

## Contact

If you have any questions, please feel free to contact [Hamza Kebiri](mailto:hamza.kebiri@unil.ch).