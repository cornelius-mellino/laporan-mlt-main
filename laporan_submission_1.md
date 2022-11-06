# Laporan Proyek Machine Learning - Cornelius Mellino Sarungu

## 1. Domain Proyek

Proyek machine learning terapan kali ini adalah membuat satu pemodelan yang mampu memprediksi kelayakan pemberian persetujuan kredit KPR untuk pengajuan yang diproses. Beberapa variabel atau fitur menjadi elemen dari inputan pemodelan yang dibangun.

- Eligibility atau kelayakan dalam konteks persetujuan pemberian kredit amat penting bagi institusi finansial seperti bank dan koperasi yang menjalankan usaha simpan-pinjam. Proses manual memakan waktu cukup lama, dalam hitungan hari, yang tentu saja membuat customer menunggu. Dengan solusi implementasi model machine learning diharapkan penentuan eligibilitas ini mampu dipercepat hingga hitungan menit, tentu saja dengan mempertimbangkan fitur-fitur yang esensial. Dalam pemodelan yang dilakukan kali ini, digunakan algoritma logistic regression yang mampu memberikan output yang mewakili keputusan disetujui (1) atau ditolak (0). 
  
## 2. Business Understanding

Dalam pengajuan kredit, baik manual (paper based) dan online, semuanya harus melalui serangkaian pengecekan data, analisa dan scoring untuk menentukan kelayakannya untuk disetujui. Hal yang menentukan kelayakan tersebut biasanya berupa serangkaian variabel yang harus diisi nilainya saat nasabah mengisi form pengajuannya. 

### 2.1. Problem Statements

Poin-poin masalah yang terdapat di proses pengajuan kredit antara lain:
- Form panjang, banyak sekali kolom (variabel) yang harus diisi oleh nasabah.
- Pemrosesan kredit lama.

### 2.2. Goals

Tujuan dari implementasi solusi machine learning ini antara lain:
- Memanfaatkan variabel atau fitur yang benar-benar memiliki keterkaitan esensial saja dengan hasil kelayakan pengajuan kredit.
- Mempercepat pengajuan kredit dengan mengimplementasikan pemodelan machine learning berbasiskan algoritma logistic regression.

### 2.3. Solution specifications
  Implementasi pemodelan machine learning untuk memprediksi nilai kelayakan pengajuan kredit kali ini memiliki spesifikasi sebagai berikut:
  - Model menggunakan algoritma Logistic Regression.
  - Penilaian performa menggunakan beberapa buah metrik/metode pengukuran.

    a. **Mean Squared Error (MSE)** 
    
    - Metrik ini mengkuadratkan perbedaan nilai antara prediksi dan aktual, lalu mengambil nilai akhir rata-ratanya.

    b. **Confusion Matrix** 
    
    - Matrix ini memetakan hasil prediksi ke dalam beberapa kategori, antara lain:

       - True Positive - nilai prediksi 1, nilai aktual 1.
       - True Negative - nilai prediksi 0, nilai aktual 0.
       - False Positive - nilai prediksi 1, nilai aktual 0
       - False Negative - nilai prediksi 0, nilai aktual 1
       
       **Akurasi** 
       
       - Akurasi diukur dengan rumus = $\frac{(TP + TN)}{(TP + TN + FP + FN)}$
       
       **Presisi** 
       
       - Presisi diukur dengan rumus = $\frac{TP}{(TP + FP)}$
       
       **Sensitivitas / Recall** 
       
       - Sensitivitas diukur dengan rumus = $\frac{TP}{(TP + FN)}$
    
    c. **Area Under Curve (AUC)** 
    
    - Area dibawah kurva (area under the curve) atau yang disebut juga dengan auc dipakai sebagai ukuran untuk menilai baik atau buruknya suatu model. AUC mendekati 1 berarti bahwa model tersebut memiliki performa baik, sedangkan AUC mendekati 0.5 menandakan bahwa model memiliki performa buruk. Kurva disini adalah kurva ROC.

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

  1. Loan Eligibility Dataset. 2020. (https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset)
  2. Loan Eligibility Machine Learning. 2020. (https://www.kaggle.com/code/vikasukani/loan-eligibility-prediction-machine-learning)
  3. Pengenalan Machine Learning dengan Python, Dios Kurniawan, M.Sc, Elex Media, Jakarta, 2021.
  4. Scikit Learn Logistic Regression. (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  5. Understanding The 3 Most Common Loss Functions for Machine Learning Regression. 2019. (https://towardsdatascience.com/understanding-the-3-most-common-loss-functions-for-machine-learning-regression-23e0ef3e14d3)
  6. Advantages and Disadvantage of Logistic Regression. 2022. (https://www.geeksforgeeks.org/advantages-and-disadvantages-of-logistic-regression/)
  7. Why Data Scaling is Important in Machine Learning. 2021. (https://analyticsindiamag.com/why-data-scaling-is-important-in-machine-learning-how-to-effectively-do-it/#:~:text=So%20if%20the%20data%20in,between%20them%20will%20be%20lower.)
  8. Random Forest Pros and Cons. 2020. (https://medium.datadriveninvestor.com/random-forest-pros-and-cons-c1c42fb64f04)
  9. Boosting. 2022. (https://corporatefinanceinstitute.com/resources/data-science/boosting/)