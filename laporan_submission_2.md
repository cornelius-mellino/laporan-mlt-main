# Laporan Proyek Machine Learning Terapan - Cornelius Mellino Sarungu
# Topik 2: Recommender System

## 1. Domain Proyek

Proyek *machine learning* terapan kali ini adalah membuat satu *recommender system* yang dapat memberi rekomendasi film kepada para user dari sebuah penyedia konten film yang berbasis *subscription*. Sistem pemberi rekomendasi adalah sistem yang membantu pengguna menemukan item yang mungkin mereka sukai [1]. Pada sebuah penyedia konten film atau video, implementasi *recommender system* akan sangat membantu pengguna dalam menentukan pilihan konten berikutnya yang akan langsung ditonton atau disimpan ke dalam *watch later* dan ditonton kemudian. Rekomendasi yang diberikan tentu saja harus memperhatikan beberapa faktor, antara lain preferensi pengguna. Indikator-indikator preferensi ini dapat dibaca dari aspek-aspek seperti genre yang menjadi favorit, rating yang diberikan, durasi tonton dan sebagainya.

Netflix, sebagai salah satu *provider* konten film yang cukup besar, menerapkan *recommender system* sebagai salah satu ujung tombak di dalam layanannya [2]. Di satu bagian situs web-nya, Netflix menjelaskan bahwa mereka memperkirakan pilihan film-film berikutnya dari pengguna dengan menggunakan beberapa faktor sebagai acuan, seperti berikut:
   * Riwayat menonton pengguna.
   * Riwayat penilaian pengguna terhadap film-film yang ada.
   * Data pengguna lain yang memiliki preferensi serupa.
   * Informasi judul, genre, kategori, aktor, tahun rilis.

Mengapa *recommender system* penting bagi bisnis, terutama yang menawarkan spektrum ragam produk yang luas? Kita perlu memahami bahwa dalam samudera informasi yang berlebihan ini, sebuah *filter* yang membantu kita (baca: *customer*) dalam menemukan informasi yang tepat dan sesuai dengan keinginan kita adalah sebuah alat bantu yang luar biasa. Hal itu akan sangat membantu kita mempersingkat waktu pencarian. Selain itu dari sisi penyedia informasi, semakin cepatnya *customer* menemukan item yang tepat sesuai keinginannya, semakin cepat pula mereka merealisasikan transaksi, dan itu artinya semakin cepat pula *revenue* mengalir masuk dan *profit* didapatkan [3].

## 2. Business Understanding

### 2.1. Problem Statements

Poin-poin masalah yang muncul di penyedia layanan konten film antara lain:

- Masifnya jumlah konten yang ada dalam database penyedia layanan konten film dapat membuat *customer* mengalami *information overwhelmed* ketika mencari apa yang mereka butuhkan. Bila hal ini muncul, menurut Khasada et al. (2020), efek yang akan muncul antara lain adalah [4]:
   * Turunnya kinerja dalam mengerjakan sesuatu.
   * Semakin lamanya proses pengambilan keputusan secara individual.
   * Menaikkan tingkat *stress*.
   * Membawa dampak negatif secara psikologis bagi individu.

Tentu saja apabila hal ini sampai terjadi pada *customer* sebuah bisnis, maka dampaknya akan luar biasa negatif. Semua itu akan bermuara pada menurunnya pemakaian oleh pengguna, menurunnya jumlah pengguna, tidak adanya lagi pengguna baru, menurunnya transaksi, yang akan menurunkan tingkat penjualan (*sales*), dan artinya penurunan *revenue* serta *profit* bagi perusahaan.

- Pengambilan keputusan oleh *customer* yang menghadapi *information overwhelm* akan sangat lambat karena yang bersangkutan akan menghabiskan banyak waktu memilih produk dan item yang diinginkan dari sebegitu banyak item produk yang ada. *Search engine* biasa memang sangat membantu, namun *search engine* tidak dapat merekomendasikan item produk yang terbaik, konten film yang paling sesuai selera, yang paling mendekati preferensi *customer*. Tetap saja *customer* perlu mengalokasikan waktu yang cukup untuk memeriksa satu demi satu hasil pencarian dari *search engine* untuk menentukan mana item produk yang relevan dan tidak.

- Semakin lama pengambilan keputusan oleh *customer*, mengakibatkan semakin tertundanya realisasi transaksi. Dampaknya adalah melambatnya *revenue stream* perusahaan. Bila pengambilan keputusan ini bisa dipercepat, diharapkan aliran *revenue stream* juga akan semakin meningkat kecepatannya.

### 2.2. Goals

Tujuan dari implementasi solusi *recommender system* ini antara lain:

- Mengurangi beban *customer* dalam melakukan pencarian produk yang relevan dengan preferensi mereka. Diharapkan dengan penerapan *recommender system* ini *customer* mendapatkan *tool* yang sangat membantu dalam melakukan pemilihan item produk yang sesuai keinginan mereka dengan cepat dan kepuasan mereka dalam memakai aplikasi penyedia konten film ini dapat meningkat lagi.

- Mempercepat proses pengambilan keputusan oleh *customer*, sehingga realisasi transaksi dapat terjadi lebih cepat lagi.

- Meningkatnya proses pengambilan keputusan dalam merealisasikan transaksi akan mempercepat aliran *revenue stream* perusahaan.

### 2.3. Solution specifications
  Implementasi pemodelan *recommender system* untuk menghasilkan rekomendasi konten yang relevan sesuai preferensi *customer* ini memiliki spesifikasi sebagai berikut:
  
  - Mengimplementasikan solusi *recommendation system* berjenis *user based collaborative recommender system*.
  
  - *Recommender system* menerima *input* user ID.
  
  - *Recommender system* menghasilkan daftar film yang direkomendasikan bagi user yang user ID nya dimasukkan sebagai *input*.

  -  Algoritma yang dipakai untuk menghitung *similarity* adalah Euclidian Distance.


## 3. Data Understanding
Data yang dipakai pada proyek ini adalah *Movie Dataset* dari Kaggle oleh Shinigami (https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset) [5].

*Movie dataset* ini terdiri dari dua buah file .csv, yaitu movies.csv dan ratings.csv. Pada movies.csv berisi data tentang film yang jumlahnya sekitar 9737 buah, lengkap dengan klasifikasi genrenya. Sedangkan pada ratings.csv berisi data *rating* yang diberikan oleh sekitar 610 *user* dengan jumlah total rating yang diberikan sekitar 100.000-an.

Untuk eksperimen ini kita akan memakai sekitar 5000 data saja untuk menguji validitas dari algoritma yang akan kita terapkan.

```python
movies = movies[:5000]
 ```

### 3.1. Struktur data Movie
Untuk struktur data dari dataframe movies adalah sebagai berikut:

| # | Column  | Non-Null Count | Dtype  |
|---|---------|----------------|--------| 
| 0 | movieId | 9742 non-null  | int64  |
| 1 | title   | 9742 non-null  | object |
| 2 | genres  | 9742 non-null  | object |

### 3.2. Struktur data Rating

| # | Column    | Non-Null Count  | Dtype   |
|---|-----------|-----------------|---------|  
| 0 | userId    | 100836 non-null | int64   |
| 1 | movieId   | 100836 non-null | int64   |
| 2 | rating    | 100836 non-null | float64 |
| 3 | timestamp | 100836 non-null | int64   |

### 3.3. Porsi rating dalam keseluruhan Movies
Dari diagram berikut kita dapat ketahui bahwa movies dengan nilai rating 4.0 memiliki porsi paling banyak di dalam database, yaitu sekitar 24%.

|[<img src="/assets/images/movie1.png"/>](/assets/images/movie1.png)|
|:--:| 
| *Gambar 1. Porsi rating dalam keseluruhan movies.* |

### 3.4. Porsi rating dalam keseluruhan Users
Dari diagram berikut kita dapat ketahui bahwa user paling banyak memberikan rating 4.0, yaitu sekitar 26%, kemudian disusul rating 3.0 dan 3.5, masing-masing 20% dan 13%.

|[<img src="/assets/images/movie2.png"/>](/assets/images/movie2.png)|
|:--:| 
| *Gambar 1. Porsi rating dalam keseluruhan users.* |


## 4. Data Preparation
Teknik *data preparation* yang dilakukan untuk mempersiapkan data sebelum diproses ke dalam model machine learning antara lain:




## 5. Modeling


## 6. Evaluation
Evaluasi kinerja *recommender system* ini dapat ditinjau dari beberapa sisi. Berikut ini adalah aspek-aspek yang menjadi pertimbangan evaluasi kinerja:
   1. *Processing speed*
      Kecepatan proses memiliki poin penting dalam sebuah *recommender system*, apalagi yang berurusan dengan jumlah data yang besar dan tentu saja bertambah terus ukurannya dari waktu ke waktu. Maka dari itu baik algoritma berbasis perhitungan aljabar, statistik maupun *neural network* harus mempertimbangkan hal ini.

      Pada eksperimen kali ini dengan ukuran data sebagai berikut:
      * User = 66
      * Movie = 2000
      * 




## 7. Kesimpulan



## 8. Referensi:

  [1] M. Longo, “The what, why and how of Recommendation Systems,” Medium, 18-Apr-2018. [Online]. Available: https://medium.com/retargetly/the-what-why-and-how-of-recommendation-systems-810d98789f83. [Accessed: 10-Feb-2023].
  
  [2] Netflix, “How netflix's recommendations System Works,” Help Center. [Online]. Available: https://help.netflix.com/en/node/100639. [Accessed: 10-Feb-2023]. 

  [3] F. O. Isinkaye, Y. O. Folajimi, and B. A. Ojokoh, “Recommendation systems: Principles, methods and evaluation,” Egyptian Informatics Journal, vol. 16, no. 3, pp. 261–273, 2015. 
  
  [4] A. Kashada, A. Isnoun, and N. Aldali, “Effect of Information Overload on Decision's Quality, Efficiency and Time,” International Journal of Latest Engineering Research and Applications (IJLERA), vol. 05, no. 1, pp. 53–58, Jan. 2020. 
  
  [5] Shinigami, “Movie recommender system dataset,” Kaggle, 02-May-2021. [Online]. Available: https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset. [Accessed: 10-Feb-2023]. 
  
  [6] John S. Breese, David Heckerman, and Carl Kadie. 1998. Empirical analysis of predictive algorithms for collaborative filtering. In Proceedings of the Fourteenth conference on Uncertainty in artificial intelligence (UAI'98). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 43–52.