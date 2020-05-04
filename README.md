# DeepHINT
Deep learning for HIV 

Install scikit-learn (http://scikit-learn.org/), keras 2.0.8 (https://keras.io/) and bedtools (http://bedtools.readthedocs.io/en/latest/) first.

All the trained deep learning models are in the model/ folder. To reproduce the result in the data, 1) Dowload the test bed file from data/ folder, 2) download the hg19 reference genome from UCSC, 3) run data.py (mofidy the bedtools command in the script) to generate test inputs (X) and labels (y), 4) run DeepHINT.py 

Note that a user-friendly online version of DeepHINT is coming soon. The current implementation of DeepHINT was based in part on the source code of iDeepA (https://github.com/xypan1232/iDeepA).

If you have any questions, please feel free to contact me : )

Email: huhailin92@gmail.com
