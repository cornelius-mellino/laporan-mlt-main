# Laporan Proyek Machine Learning - Cornelius Mellino Sarungu

## 1. Domain Proyek

Proyek machine learning terapan kali ini adalah membuat satu pemodelan yang mampu memprediksi kelayakan pemberian persetujuan kredit pemilikan rumah (KPR) untuk pengajuan yang diproses. Beberapa variabel atau fitur menjadi elemen dari inputan pemodelan yang dibangun.

- Proses kredit adalah proses meminjamkan sejumlah dana kepada seseorang atau lembaga supaya mereka dapat memenuhi kebutuhannya dengan dana tersebut. Untuk kredit yang mentarget perseorangan disebut juga sebagai kredit retail, sedangkan yang mentargetkan perusahaan atau lembaga disebut sebagai kredit korporasi. Dalam pengajuannya, terdapat banyak hal yang perlu dipertimbangkan. Bahkan untuk beberapa jenis kredit membutuhkan agunan / jaminan berupa aset yang variasinya juga beragam. Untuk kredit pemilikan rumah (KPR) misalnya, rumah yang kita cicil akan secara otomatis menjadi jaminan dan sertifikatnya dipegang oleh pihak pemberi kredit sampai cicilan tersebut lunas (Suyatno, 2007). 

- Namun baik pengajuan kredit retail maupun korporasi, keduanya harus melewati serangkaian proses sebelum sampai ke tahap persetujuan. Rangkaian proses tersebut diantaranya antara lain pemeriksaan latar belakang, pemeriksaan skor kelayakan berdasarkan variabel-variabel yang didapat informasinya dari form yang diisi oleh nasabah, pemeriksaan skor kelayakan dari pihak ketiga seperti Pefindo untuk di Indonesia, pemeriksaan kelengkapan dokumen, validasi biometrik. Setelah semua pemeriksaan tadi menghasilkan nilai positif, maka pengajuan kredit dapat ditetapkan sebagai layak atau eligible untuk dipenuhi (Hasibuan, 2008).

## 2. Business Understanding

Dalam pengajuan kredit, baik manual (paper based) dan online, semuanya harus melalui serangkaian pengecekan data, analisa dan scoring untuk menentukan kelayakannya untuk disetujui. Hal yang menentukan kelayakan tersebut biasanya berupa serangkaian variabel yang harus diisi nilainya saat nasabah mengisi form pengajuannya. 

- Dalam melakukan sebuah pengajuan kredit, nasabah harus mengisi form pengajuan yang biasanya memiliki banyak kolom yang harus diisi. Form pengajuan kredit ini isinya meliputi: bio data, informasi tempat tinggal, informasi pekerjaan, informasi tempat kerja, informasi finansial (hutang, aset), informasi kontak terdekat, informasi perbankan (nomor rekening, kartu kredit), dokumen pelengkap. Sedangkan untuk kredit korporasi, biasanya harus mengajukan proposal kredit yang berisi kurang lebih informasi sebagai berikut: executive summary perusahaan, identitas dan struktur perusahaan, gambaran umum perusahaan, kondisi keuangan perusahaan, analisis industri, struktur  keuangan perusahaan, analisis proyeksi keuangan, jaminan kredit, lampiran (Jusuf, 2003).

- Eligibility atau kelayakan dalam konteks persetujuan pemberian kredit amat penting bagi institusi finansial seperti bank dan koperasi yang menjalankan usaha simpan-pinjam. Proses manual memakan waktu cukup lama, dalam hitungan hari, yang tentu saja membuat customer menunggu. Persaingan penyedia layanan kredit online membuat para pemainnya berlomba-lomba memaksimalkan layanan bisnis mereka terutama pada peningkatan kecepatan proses persetujuan. Perlombaan di aspek ini memaksa para pemain tersebut melakukan eksplorasi teknologi machine learning dan AI untuk mempersingkat waktu, dan dalam waktu yang bersamaan mendapatkan akurasi yang tinggi akan keputusan persetujuan yang diambil. Akurasi yang tinggi diperlukan, karena kesalahan dalam pemberian keputusan kredit akan menimbulkan kerugian bagi penyedia jasa layanan kredit terkait (Amrin & Pahlevi, 2022). Baik salah menyetujui nasabah yang tidak layak diberi kredit (false positive), maupun tidak menyetujui nasabah yang layak diberi kredit (false negative), semuanya sama-sama membuka potensi kerugian (Amrin & Pahlevi, 2022). Dengan solusi implementasi model machine learning diharapkan penentuan eligibilitas ini mampu dipercepat hingga hitungan menit, tentu saja dengan mempertimbangkan fitur-fitur yang esensial. Dalam pemodelan yang dilakukan kali ini, dicoba untuk mengimplementasikan beberapa algoritma untuk pemodelan antara lain: logistic regression, random forest dan boosting yang semuanya mampu memberikan output yang mewakili keputusan disetujui (1) atau ditolak (0).

### 2.1. Problem Statements

Poin-poin masalah yang terdapat di proses pengajuan kredit antara lain:

- Pemrosesan form pengajuan kredit yang membutuhkan waktu yang cukup lama. Lamanya waktu proses semenjak form disubmit sampai diterimanya notifikasi disetujui / ditolak oleh nasabah menjadi salah satu titik penting dalam persaingan layanan pemberian kredit. Dari sisi nasabah semakin cepat tentu saja akan semakin menaikkan tingkat kepuasan terhadap layanan tersebut. Sedangkan dari sisi institusi finansial, meningkatnya kecepatan akan menaikkan nilai competitiveness dari layanannya di arena bisnis terkait.

- Akurasi keputusan persetujuan pemberian kredit menjadi hal yang sangat penting bagi institusi finansial penyedia layanan tersebut. Kesalahan dalam pemberian persetujuan akan membawa kerugian finansial bagi institusi tersebut. Bila ternyata kredit disetujui namun nasabah tidak mampu mengembalikan maka akan membawa kerugian finansial yang nyata, bila kredit tidak disetujui padahal nasabah seharusnya layak untuk diberi dan memiliki kemampuan untuk mengembalikan pinjamannya maka akan membawa kerugian berupa loss of opportunity bagi institusi penyedia layanan kredit.

### 2.2. Goals

Tujuan dari implementasi solusi machine learning ini antara lain:

- Mempercepat pengajuan kredit dengan mengimplementasikan pemodelan machine learning berbasiskan beberapa algoritma pemodelan, antara lain logistic regression, random forest dan boosting. Ketiga algoritma tersebut akan diperbandingkan kinerjanya dan akan diulas kelebihan maupun kekurangannya.

- Mencari algoritma pemodelan yang memiliki tingkat akurasi paling tinggi, yang dapat membantu institusi pemberi layanan kredit dalam pemberian persetujuan kepada pengajuan yang diajukan oleh para nasabah. Hal ini juga tentunya akan membantu menekan risiko kerugian yang mungkin terjadi di masa depan nantinya.

### 2.3. Solution specifications
  Implementasi pemodelan machine learning untuk memprediksi nilai kelayakan pengajuan kredit kali ini memiliki spesifikasi sebagai berikut:
  
  - Mengimplementasikan pemodelan menggunakan tiga buah algoritma: Logistic Regression, Random Forest dan Boosting.
  
  - Penilaian performa terhadap ketiga pemodelan yang akan dibuat menggunakan beberapa buah metrik/metode pengukuran, antara lain sebagai berikut:

    a. **Mean Squared Error (MSE)** 
    
    - Metrik ini mengkuadratkan perbedaan nilai antara prediksi dan aktual, lalu mengambil nilai akhir rata-ratanya (Bickel, 2015).

        - Rumus MSE adalah sebagai berikut:

            $MSE = \frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$

    b. **Confusion Matrix** 
    
    - Matrix ini memetakan hasil prediksi ke dalam beberapa kategori, antara lain:

       - True Positive - nilai prediksi 1, nilai aktual 1.
       - True Negative - nilai prediksi 0, nilai aktual 0.
       - False Positive - nilai prediksi 1, nilai aktual 0
       - False Negative - nilai prediksi 0, nilai aktual 1
       
       **Akurasi** 
       
       - Akurasi diukur dengan rumus berikut:

          $accuracy = \frac{(TP + TN)}{(TP + TN + FP + FN)}$
       
       **Presisi** 
       
       - Presisi diukur dengan rumus berikut:
       
          $precision = \frac{TP}{(TP + FP)}$
       
       **Sensitivitas / Recall** 
       
       - Sensitivitas diukur dengan rumus berikut:
       
          $sensitivity = \frac{TP}{(TP + FN)}$
    
    c. **Area Under Curve (AUC)** 
    
    - Area dibawah kurva (area under the curve) atau yang disebut juga dengan auc dipakai sebagai ukuran untuk menilai baik atau buruknya suatu model. AUC mendekati 1 berarti bahwa model tersebut memiliki performa baik, sedangkan AUC mendekati 0.5 menandakan bahwa model memiliki performa buruk. Kurva disini adalah kurva ROC.

      |[<img src="/assets/images/roc_curve.png" height="300" width="300"/>](/assets/images/roc_curve.png)|
      |:--:| 
      | *Gambar 1. Kurva ROC.* |

## 3. Data Understanding
Data yang dipakai pada proyek ini adalah Loan Eligible Dataset dari Kaggle oleh Vikas Ukani (https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset).

Data tersebut adalah data dari perusahaan Dream Housing Finance yang menangani semua pinjaman KPR. Mereka hadir di semua wilayah perkotaan, semi-perkotaan, dan pedesaan. Pelanggan pertama-tama mengajukan pinjaman KPR setelah itu perusahaan memvalidasi kelayakan pelanggan untuk pinjaman.

### 3.1. Variabel-variabel pada Loan Eligible Dataset adalah sebagai berikut:
- Loan_ID	: ID pinjaman, unique code / number
- Gender : Jenis kelamin,	Male/ Female
- Married	: Status menikah dari applicant, (Y/N)
- Dependents : Jumlah tanggugan, number
- Education : Pendidikan terakhir applicant, (Graduate/ Under Graduate)
- Self_Employed	: Apakah applicant self-employed, (Y/N)
- ApplicantIncome	: Pendapatan applicant, number
- CoapplicantIncome	: Pendapatan istri, number
- LoanAmount : Jumlah pinjaman, number dalam ribuan
- Loan_Amount_Term : Jangka waktu pinjaman, number dalam bulan
- Credit_History : Sejarah pinjaman, ada atau tidak (1/0)
- Property_Area	: Area properti, Urban/ Semi-Urban/ Rural
- Loan_Status	: Status persetujuan kredit, (Y/N)

Setelah data mentah diload, kita melakukan serangkaian aktivitas exploratory sebagai berikut:
- Melihat bagian awal tabel data dengan function head().
- Melihat summary data dengan function describe().
- Melihat struktur data dengan function info().

## 4. Data Preparation
Teknik data preparation yang dilakukan untuk mempersiapkan data sebelum diproses ke dalam model machine learning antara lain:

- Mengidentifikasi nilai null.
- Mengganti nilai null dengan nilai rerata atau modus.
- Mengubah nilai kategorikal menjadi nilai numerikal.
- Membuat visualisasi heatmap untuk memeriksa keterkaitan antar fitur.
- Memeriksa masing-masing fitur dengan visualisasi histogram atau scatter plot.

## 5. Modeling
### 5.1. Logistic Regression
Pemodelan pertama yang dipilih untuk solusi ini adalah model yang menggunakan algoritma Logistic Regression, karena algoritma ini cocok untuk permasalahan dengan banyak variabel independen dan menghasilkan output biner (0/1, Ya/Tidak, Approve/Reject dll).

**Kelebihan**

- Mudah diimplementasikan.
- Bisa mengakomodir multi-variabel.
- Tidak hanya memberikan ukuran seberapa tepat suatu prediktor (ukuran koefisien), tetapi juga arah asosiasinya (positif atau negatif).
- Sangat cepat dalam mengklasifikasikan record-record yang tidak diketahui.
- Memiliki tingkat akurasi yang baik untuk kumpulan data sederhana dan berkinerja baik ketika kumpulan data dapat dipisahkan secara linier.
- Ini dapat menginterpretasikan koefisien model sebagai indikator pentingnya fitur.

**Kekurangan**

- Jika jumlah observasi lebih kecil dari jumlah fitur, Logistic Regression tidak bisa dipakai karena dapat menyebabkan overfitting.
- Keterbatasan utama Regresi Logistik adalah asumsi linearitas antara variabel dependen dan variabel independen.
- Hanya dapat digunakan untuk memprediksi fungsi diskrit. Oleh karena itu, variabel terikat Regresi Logistik terikat pada himpunan bilangan diskrit.
- Masalah non-linier tidak dapat diselesaikan dengan regresi logistik karena memiliki permukaan keputusan linier. Data yang dapat dipisahkan secara linier jarang ditemukan dalam skenario dunia nyata.
- Regresi Logistik membutuhkan rata-rata atau tidak ada multikolinearitas antar variabel bebas.
- Sulit untuk mendapatkan hubungan yang kompleks menggunakan regresi logistik. Algoritme yang lebih kuat dan ringkas seperti Neural Networks dapat dengan mudah mengungguli algoritme ini.

Berikut code dari model Logistic Regression dalam Python.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), LogisticRegression(solver = "lbfgs"))
pipe.fit(X_train, y_train)  # apply scaling on training data
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('logisticregression', LogisticRegression(solver = "lbfgs"))])

pipe.score(X_val, y_val) 
```
Pada code diatas, kita menggunakan teknik pipelining dimana sebelum data masuk ke model LogisticRegression(), data di-scaling terlebih dahulu dengan metode standar. Hal ini agar data memiliki memiliki jarak yang lebih pendek. Scaling perlu dilakukan bila nilai dari fitur-fitur yang ada memiliki rentang yang teramat jauh satu sama lain. Terkadang bila kita tidak melakukan scaling, model kita akan mengalami crash secara komputasional.

### 5.2. Random Forest
Pemodelan kedua yang dipilih adalah Random Forest. Algoritma ini merupakan salah satu algoritma Ensemble dimana konsepnya adalah mengkombinasikan hasil dari beberapa algoritma yang dijalankan untuk mencari hasil yang optimal. Di Random Forest, beberapa Decision Trees dibuat kemudian untuk masalah klasifikasi, kelas-kelas yang banyak terpilih oleh semua Decision Trees akan dipilih, sedangkan untuk masalah regresi dipilih nilai rata-rata outputnya. Algoritma ini cenderung lebih baik daripada algoritma Decision Trees tunggal, dan mampu memperbaiki masalah overfittingnya juga.

**Kelebihan**

- Random Forest dapat digunakan untuk tugas klasifikasi dan regresi.
- Random Forest bekerja dengan baik dengan data kategorikal dan numerik. Biasanya tidak diperlukan penskalaan atau transformasi variabel.
- Random Forest secara implisit melakukan pemilihan fitur dan menghasilkan Decision Trees yang tidak berkorelasi. Ini dilakukan dengan memilih serangkaian fitur acak untuk membangun setiap Decision Trees. Ini juga menjadikannya model yang hebat ketika Anda harus bekerja dengan sejumlah besar fitur dalam data.
- Random Forest tidak dipengaruhi oleh outlier sampai tingkat yang wajar. Ini dilakukan dengan menggabungkan variabel.
- Random Forest dapat menangani hubungan linier dan non-linier dengan baik.
- Random Forest umumnya memberikan akurasi tinggi dan menyeimbangkan trade-off bias-varians dengan baik. Karena prinsip model adalah untuk merata-ratakan hasil di beberapa Decision Trees yang dibuatnya, model ini juga meratakan variansnya.

**Kekurangan**

- Random Forest tidak mudah ditafsirkan. Mereka memberikan kepentingan fitur tetapi tidak memberikan visibilitas lengkap ke dalam koefisien sebagai regresi linier.
- Random Forest dapat secara komputasi intensif untuk kumpulan data besar.
- Random Forest seperti algoritma black box, Anda hanya memiliki sedikit kendali atas apa yang dilakukan model.

(Jagandeep Singh, 2020)

Berikut code dari model Random Forest dalam Python.

```python
from sklearn.ensemble import RandomForestRegressor
 
# buat model prediksi
RF = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
 
models.loc['train_mse','Random Forest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)       
models.loc['test_mse','Random Forest'] = mean_squared_error(y_pred=RF.predict(X_val), y_true=y_val) 
```

### 5.3. Boosting 
Pemodelan ketiga yang dipilih adalah Boosting. Algoritma ini mengkombinasikan beberapa algoritma "learners" yang ujungnya diharapkan mampu menghasilkan keluaran yang kuat. Di dalam algoritma Boosting ini dikombinasikan beberapa algoritma yang masing-masing punya kelebihan dan kekurangan, dimana diharapkan kombinasi antara "learners" yang lemah dan kuat akan menghasilkan keluaran prediksi yang lebih tepat.

**Kelebihan**

- Boosting hadir dengan algoritme yang mudah dibaca dan ditafsirkan, membuat interpretasi prediksinya mudah ditangani.
- Kemampuan prediksinya efisien melalui penggunaan metode kloningnya, seperti bagging atau Random Forest dan Decision Trees. 
- Boosting adalah metode cukup handal untuk mengantisipasi over-fitting dengan mudah.

**Kekurangan**

- Metode ini sensitif terhadap outlier karena setiap classifier berkewajiban untuk memperbaiki kesalahan pada pendahulunya.
- Kerugian lain adalah bahwa metode ini hampir tidak mungkin untuk ditingkatkan. Hal ini karena setiap estimator mendasarkan kebenarannya pada prediktor sebelumnya, sehingga membuatnya sulit untuk disederhanakan.

(CFI Team, 2022)

Berikut code dari model Boosting dalam Python.

```python
from sklearn.ensemble import AdaBoostRegressor
 
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)                             
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
models.loc['test_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_val), y_true=y_val)
```

## 6. Evaluation
Evaluasi kinerja pemodelan machine learning dilakukan dengan beberapa cara.

Pada tahapan evaluasi ini pemodelan dengan menggunakan Logistic Regression diukur kinerjanya dengan beberapa metriks. Berikut ini penjelasannya:

### 6.1. **Mean Squared Error (MSE)** 
  
Kelebihan MSE disini salah satunya adalah untuk mengidentifikasi apakah di model kita terdapat outlier yang akan menyebabkan nilai error menjadi sangat besar. Nilai MSE yang cukup besar (> 0.1) disini kemungkinan disebabkan adanya rentang yang cukup jauh antara nilai rata-rata dengan maksimum di beberapa fitur seperti pendapatan dan pendapatan pasangan.

Berikut hasil evaluasi MSE dari ketiga model:

|           | Logistic Regression | Random Forest | Boosting |
|-----------|---------------------|---------------|----------|
| train_mse | 0.188797            | 0.029990      | 0.13790  |
| test_mse  | 0.165289            | 0.153723      | 0.14803  |

### 6.2.a. **Confusion Matrix**

- Matrix ini memetakan hasil prediksi ke dalam beberapa kategori, antara lain:
       
|                | Nilai Prediksi | Nilai Aktual |
|----------------|----------------|--------------|
| True Positive  | 1              | 1            |
| False Positive | 0              | 0            |
| False Negative | 1              | 0            |
| True Negative  | 0              | 1            |

- Hasil pembuatan confusion matrix dari perbandingan antara keluaran riil validasi (y_val) dengan keluaran prediktif dari model (y_pred) adalah sebagai berikut:

|                  | Logistic Regression | Random Forest     | Boosting          |
|------------------|---------------------|-------------------|-------------------|
| Confusion Matrix | [16, 19]            | [19, 16]          | [16, 19]          |
|                  | [1 , 85]            | [12, 74]          | [1 , 85]          |
| True Positive    | 16                  | 19                | 16                |
| False Positive   | 19                  | 16                | 19                |
| False Negative   | 1                   | 12                | 1                 |
| True Negative    | 85                  | 74                | 85                |

### 6.2.b. **Akurasi** 
  
- Akurasi diukur dengan rumus = $$\frac{(TP + TN)}{(TP + TN + FP + FN)}$$

|                     | Accuracy           |
|---------------------|--------------------|
| Logistic Regression | 0.8347107438016529 |
| Random Forest       | 0.768595041322314  |
| Boosting            | 0.8347107438016529 | 

### 6.2.c. **Presisi**

- Presisi diukur dengan rumus = $$\frac{TP}{(TP + FP)}$$
  
|                     | Precision          |
|---------------------|--------------------|
| Logistic Regression | 0.8173076923076923 |
| Random Forest       | 0.8222222222222222 |
| Boosting            | 0.8173076923076923 |

### 6.2.d. **Sensitivitas / Recall** 

- Sensitivitas diukur dengan rumus = $$\frac{TP}{(TP + FN)}$$
  
|                     | Sensitivity        |
|---------------------|--------------------|
| Logistic Regression | 0.9883720930232558 |
| Random Forest       | 0.8604651162790697 |
| Boosting            | 0.9883720930232558 |

## 6.3. **Area Under Curve (AUC)** 

- Area dibawah kurva (area under the curve) atau yang disebut juga dengan auc dipakai sebagai ukuran untuk menilai baik atau buruknya suatu model. AUC mendekati 1 berarti bahwa model tersebut memiliki performa baik, sedangkan AUC mendekati 0.5 menandakan bahwa model memiliki performa buruk. Kurva disini adalah kurva ROC (Receiver Operating Characteristics).

|                     | AUC                |
|---------------------|--------------------|
| Logistic Regression | 0.7227574750830565 |
| Random Forest       | 0.7016611295681063 |
| Boosting            | 0.7227574750830565 |

## 7. Kesimpulan

Kesimpulan dari perbandingan evaluasi kinerja dari ketiga model (Logistic Regression, Random Forest dan Boosting) adalah bahwa algoritma Boosting memiliki kinerja yang paling optimal diantara ketiganya. Dengan skor akurasi, presisi dan sensitivitas serta AUC yang sama dengan Logistic Regression, Boosting menghasilkan nilai MSE yang lebih kecil. Sedangkan Random Forest kinerjanya masih berada dibawah dari kedua model lainnya.

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