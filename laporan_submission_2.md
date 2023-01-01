# Laporan Proyek Machine Learning - Cornelius Mellino Sarungu

## 1. Domain Proyek

Proyek *machine learning* terapan kali ini adalah membuat satu pemodelan yang mampu memprediksi kelayakan pemberian persetujuan kredit pemilikan rumah (KPR) untuk pengajuan yang diproses. Beberapa variabel atau fitur menjadi elemen dari inputan pemodelan yang dibangun.

- Proses kredit adalah proses meminjamkan sejumlah dana kepada seseorang atau lembaga supaya mereka dapat memenuhi kebutuhannya dengan dana tersebut. Untuk kredit yang mentarget perseorangan disebut juga sebagai kredit *retail*, sedangkan yang mentargetkan perusahaan atau lembaga disebut sebagai kredit korporasi. Dalam pengajuannya, terdapat banyak hal yang perlu dipertimbangkan. Bahkan untuk beberapa jenis kredit membutuhkan agunan / jaminan berupa aset yang variasinya juga beragam. Untuk kredit pemilikan rumah (KPR) misalnya, rumah yang kita cicil akan secara otomatis menjadi jaminan dan sertifikatnya dipegang oleh pihak pemberi kredit sampai cicilan tersebut lunas (Suyatno, 2007). 

- Namun baik pengajuan kredit retail maupun korporasi, keduanya harus melewati serangkaian proses sebelum sampai ke tahap persetujuan. Rangkaian proses tersebut diantaranya antara lain pemeriksaan latar belakang, pemeriksaan skor kelayakan berdasarkan variabel-variabel yang didapat informasinya dari form yang diisi oleh nasabah, pemeriksaan skor kelayakan dari pihak ketiga seperti Pefindo untuk di Indonesia, pemeriksaan kelengkapan dokumen, validasi biometrik. Setelah semua pemeriksaan tadi menghasilkan nilai positif, maka pengajuan kredit dapat ditetapkan sebagai layak atau *eligible* untuk dipenuhi (Hasibuan, 2008).

## 2. Business Understanding

Dalam pengajuan kredit, baik manual (*paper based*) dan *online*, semuanya harus melalui serangkaian pengecekan data, analisa dan *scoring* untuk menentukan kelayakannya untuk disetujui. Hal yang menentukan kelayakan tersebut biasanya berupa serangkaian variabel yang harus diisi nilainya saat nasabah mengisi form pengajuannya. 

- Dalam melakukan sebuah pengajuan kredit, nasabah harus mengisi *form* pengajuan yang biasanya memiliki banyak kolom yang harus diisi. *Form* pengajuan kredit ini isinya meliputi: bio data, informasi tempat tinggal, informasi pekerjaan, informasi tempat kerja, informasi finansial (hutang, aset), informasi kontak terdekat, informasi perbankan (nomor rekening, kartu kredit), dokumen pelengkap. Sedangkan untuk kredit korporasi, biasanya harus mengajukan proposal kredit yang berisi kurang lebih informasi sebagai berikut: *executive summary* perusahaan, identitas dan struktur perusahaan, gambaran umum perusahaan, kondisi keuangan perusahaan, analisis industri, struktur  keuangan perusahaan, analisis proyeksi keuangan, jaminan kredit, lampiran (Jusuf, 2003).

- *Eligibility* atau kelayakan dalam konteks persetujuan pemberian kredit amat penting bagi institusi finansial seperti bank dan koperasi yang menjalankan usaha simpan-pinjam. Proses manual memakan waktu cukup lama, dalam hitungan hari, yang tentu saja membuat customer menunggu. Persaingan penyedia layanan kredit online membuat para pemainnya berlomba-lomba memaksimalkan layanan bisnis mereka terutama pada peningkatan kecepatan proses persetujuan. Perlombaan di aspek ini memaksa para pemain tersebut melakukan eksplorasi teknologi *machine learning* dan *AI* untuk mempersingkat waktu, dan dalam waktu yang bersamaan mendapatkan akurasi yang tinggi akan keputusan persetujuan yang diambil. Akurasi yang tinggi diperlukan, karena kesalahan dalam pemberian keputusan kredit akan menimbulkan kerugian bagi penyedia jasa layanan kredit terkait (Amrin & Pahlevi, 2022). Baik salah menyetujui nasabah yang tidak layak diberi kredit (*false positive*), maupun tidak menyetujui nasabah yang layak diberi kredit (*false negative*), semuanya sama-sama membuka potensi kerugian (Amrin & Pahlevi, 2022). Dengan solusi implementasi model *machine learning* diharapkan penentuan eligibilitas ini mampu dipercepat hingga hitungan menit, tentu saja dengan mempertimbangkan fitur-fitur yang esensial. Dalam pemodelan yang dilakukan kali ini, dicoba untuk mengimplementasikan beberapa algoritma untuk pemodelan antara lain: *Logistic Regression*, *Random Forest* dan *Boosting* yang semuanya mampu memberikan *output* yang mewakili keputusan disetujui (1) atau ditolak (0).

### 2.1. Problem Statements

Poin-poin masalah yang terdapat di proses pengajuan kredit antara lain:

- Pemrosesan form pengajuan kredit yang membutuhkan waktu yang cukup lama. Lamanya waktu proses semenjak form disubmit sampai diterimanya notifikasi disetujui / ditolak oleh nasabah menjadi salah satu titik penting dalam persaingan layanan pemberian kredit. Dari sisi nasabah semakin cepat tentu saja akan semakin menaikkan tingkat kepuasan terhadap layanan tersebut. Sedangkan dari sisi institusi finansial, meningkatnya kecepatan akan menaikkan nilai *competitiveness* dari layanannya di arena bisnis terkait.

- Akurasi keputusan persetujuan pemberian kredit menjadi hal yang sangat penting bagi institusi finansial penyedia layanan tersebut. Kesalahan dalam pemberian persetujuan akan membawa kerugian finansial bagi institusi tersebut. Bila ternyata kredit disetujui namun nasabah tidak mampu mengembalikan maka akan membawa kerugian finansial yang nyata, bila kredit tidak disetujui padahal nasabah seharusnya layak untuk diberi dan memiliki kemampuan untuk mengembalikan pinjamannya maka akan membawa kerugian berupa *loss of opportunity* bagi institusi penyedia layanan kredit.

### 2.2. Goals

Tujuan dari implementasi solusi machine learning ini antara lain:

- Mempercepat pengajuan kredit dengan mengimplementasikan pemodelan machine learning berbasiskan beberapa algoritma pemodelan, antara lain *Logistic Regression*, *Random Forest* dan *Boosting*. Ketiga algoritma tersebut akan diperbandingkan kinerjanya dan akan diulas kelebihan maupun kekurangannya.

- Mencari algoritma pemodelan yang memiliki tingkat akurasi paling tinggi, serta memiliki nilai optimal untuk aspek penilaian lainnya yang dapat membantu institusi pemberi layanan kredit dalam pemberian persetujuan kepada pengajuan yang diajukan oleh para nasabah. Hal ini juga tentunya akan membantu menekan risiko kerugian yang mungkin terjadi di masa depan nantinya.

### 2.3. Solution specifications
  Implementasi pemodelan *machine learning* untuk memprediksi nilai kelayakan pengajuan kredit kali ini memiliki spesifikasi sebagai berikut:
  
  - Mengimplementasikan pemodelan menggunakan tiga buah algoritma: *Logistic Regression*, *Random Forest* dan *Boosting*.
  
  - Penilaian performa terhadap ketiga pemodelan yang akan dibuat menggunakan beberapa buah metrik/metode pengukuran, antara lain sebagai berikut:

    a. **Mean Squared Error (MSE)** 
    
    - Metrik ini mengkuadratkan perbedaan nilai antara prediksi dan aktual, lalu mengambil nilai akhir rata-ratanya (Bickel, 2015).

       - Rumus MSE adalah sebagai berikut:

            $MSE = \frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$

    b. **Confusion Matrix** 
    
    - Matrix ini memetakan hasil prediksi ke dalam beberapa kategori, antara lain:
       
       |                | Nilai Prediksi | Nilai Aktual |
       |----------------|----------------|--------------|
       | True Positive  | 1              | 1            |
       | False Positive | 0              | 0            |
       | False Negative | 1              | 0            |
       | True Negative  | 0              | 1            |
       
    - Berikut ini adalah pemetaan dari confusion matrix:

       | Total Poulation    | (Predicted)  Positive | (Predicted)  Negative |
       |--------------------|-----------------------|-----------------------|
       | (Actual)  Positive | True Positive (TP)    | False Negative (FN)   |
       | (Actual)  Negative | False Positive (FP)   | True Negative (TN)    |


    - **Akurasi** 
       
       - Akurasi diukur dengan rumus berikut:

          $accuracy = \frac{(TP + TN)}{(TP + TN + FP + FN)}$
       
    - **Presisi** 
       
       - Presisi diukur dengan rumus berikut:
       
          $precision = \frac{TP}{(TP + FP)}$
       
    - **Sensitivitas / Recall** 
       
       - Sensitivitas diukur dengan rumus berikut:
       
          $sensitivity = \frac{TP}{(TP + FN)}$
    
    c. **Area Under Curve (AUC)** 

    - Area dibawah kurva (area under the curve) atau yang disebut juga dengan auc dipakai sebagai ukuran untuk menilai baik atau buruknya suatu model. AUC mendekati 1 berarti bahwa model tersebut memiliki performa baik, sedangkan AUC mendekati 0.5 menandakan bahwa model memiliki performa buruk. Kurva disini adalah kurva ROC. Dari Gambar 1 dapat dilihat bahwa semakin cembung kurva ROC maka menunjukkan kinerja model semakin bagus, artinya semakin banyak hasil prediksi yang tepat. Sedangkan semakin linear kurva ROC nya, maka menunjukkan kinerja model semakin jelek. Namun perlu diketahui juga apabila nilai AUC ini terlalu mendekati 1 maka menandakan kemungkinan terjadinya overfitting di dalam pemodelan yang kita buat [2].

      |[<img src="/assets/images/roc_curve.png"/>](/assets/images/roc_curve.png)|
      |:--:| 
      | *Gambar 1. Kurva ROC. (Sumber: Wikipedia)* |

## 3. Data Understanding
Data yang dipakai pada proyek ini adalah *Loan Eligible Dataset* dari Kaggle oleh Vikas Ukani (https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset).

Data tersebut adalah data dari perusahaan Dream Housing Finance yang menangani semua pinjaman KPR. Mereka hadir di semua wilayah perkotaan, semi-perkotaan, dan pedesaan. Pelanggan pertama-tama mengajukan pinjaman KPR setelah itu perusahaan memvalidasi kelayakan pelanggan untuk pinjaman.



## 4. Data Preparation
Teknik *data preparation* yang dilakukan untuk mempersiapkan data sebelum diproses ke dalam model machine learning antara lain:




## 5. Modeling


## 6. Evaluation
Evaluasi kinerja pemodelan *machine learning* dilakukan dengan beberapa cara.




## 7. Kesimpulan

Kesimpulan dari perbandingan evaluasi kinerja dari ketiga model (*Logistic Regression*, *Random Forest* dan *Boosting*) adalah bahwa algoritma *Boosting* memiliki kinerja yang paling optimal diantara ketiganya. Dengan skor akurasi, presisi dan sensitivitas serta AUC yang sama dengan *Logistic Regression*, *Boosting* menghasilkan nilai MSE yang lebih kecil. Sedangkan *Random Forest* kinerjanya masih berada dibawah dari kedua model lainnya.

## 8. Referensi:

  [1] A. Amrin and O. Pahlevi, “Implementation of logistic regression classification algorithm and support vector machine for credit eligibility prediction,” Journal of Informatics and Telecommunication Engineering, vol. 5, no. 2, pp. 433–441, 2022. 

  [2] D. Kurniawan, Pengenalan Machine Learning dengan Python. Jakarta, Indonesia: Elex Media, 2021. 

  [3] Kasmir, Bank  dan  Lembaga  Keuangan  lainnya, Edisi Revisi. Jakarta, Indonesia: PT  Raja Grafindo Persada, 2014. 
  
  [4] L. Zhao, S. Lee, and S.-P. Jeong, “Decision tree application to classification problems with boosting algorithm,” Electronics, vol. 10, no. 16, p. 1903, Aug. 2021, doi: 10.3390/electronics10161903.

  [5] M. Gopinath, K. Srinivas Shankar Maheep, and R. Sethuraman, “Customer loan approval prediction using logistic regression,” Advances in Parallel Computing, 2021. 

  [6] M. Yarmolenko and B. Howlin, “Extreme Gradient boosting algorithm classification for predicting lifespan-extending chemical compounds,” 2022. 

  [7] M. S. P. Hasibuan, Dasar-dasar Perbankan. Jakarta, Indonesia: PT. Grafindo, 2008. 
  
  [8] J. Jusuf, Kiat Jitu Memperoleh Kredit Bank. Jakarta, Indonesia: Elex Media, 2003. 

  [9] P. J. Bickel and K. A. Doksum, Mathematical Statistics: Basic Ideas and Selected Topics, vol. 1, 2 vols. CRC Press, 2015. 

  [10] T. Suyatno, Kelembagaan  Perbankan. Jakarta, Indonesia: PT.  Gramedia Pustaka Utama, 2007. 

  [11] V. Ukani, Loan eligibility dataset, 2020. [Online]. Available: https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset. [Accessed: 04-Nov-2022]. 
  
  [12] V. Ukani, Loan eligibility machine learning, 2020. [Online]. Available: https://www.kaggle.com/code/vikasukani/loan-eligibility-prediction-machine-learning. [Accessed: 16-Oct-2022]. 