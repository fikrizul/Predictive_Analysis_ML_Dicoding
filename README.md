# Laporan Proyek Machine Learning - Fikri Zulfialdi

## Domain Proyek

Optimalisasi aktivitas latihan di pusat kebugaran telah menjadi fokus utama dalam upaya meningkatkan kesehatan masyarakat modern, terutama dalam konteks efektivitas pembakaran kalori. Pemahaman mendalam tentang aktivitas yang menghasilkan pembakaran kalori optimal tidak hanya penting bagi individu yang mengejar tujuan kebugaran pribadi, tetapi juga memiliki implikasi signifikan terhadap kesehatan masyarakat secara keseluruhan. Data dari Organisasi Kesehatan Dunia (WHO, 2020) menunjukkan bahwa tingkat aktivitas fisik yang memadai dapat secara substansial mengurangi risiko berbagai penyakit kronis, namun mayoritas populasi global masih belum mencapai tingkat aktivitas fisik yang direkomendasikan. Tantangan ini semakin diperumit oleh keterbatasan waktu yang dihadapi masyarakat modern, mendorong kebutuhan akan pemahaman yang lebih baik tentang efektivitas berbagai jenis latihan dalam memaksimalkan pembakaran kalori dalam waktu yang tersedia.

Penelitian yang dilakukan oleh Gough et al. (2018) mengungkapkan adanya pergeseran signifikan dalam preferensi masyarakat terhadap program latihan yang lebih efisien namun tetap efektif, mencerminkan kebutuhan akan optimalisasi waktu dalam konteks kesehatan modern. Tren ini memperkuat pentingnya mengidentifikasi dan memahami aktivitas gym yang memberikan manfaat maksimal dalam durasi minimal, memungkinkan individu untuk mencapai tujuan kesehatan mereka meskipun menghadapi kendala waktu. Lebih lanjut, McAuley et al. (2011) menekankan peran krusial faktor psikososial dalam efektivitas latihan, menunjukkan bahwa dukungan sosial dan kepercayaan diri secara signifikan mempengaruhi intensitas dan konsistensi latihan. Temuan ini menggarisbawahi pentingnya mempertimbangkan tidak hanya aspek fisiologis dari pembakaran kalori, tetapi juga konteks psikologis dan sosial yang mempengaruhi efektivitas latihan secara keseluruhan.

Dalam konteks kesehatan masyarakat yang lebih luas, pemahaman tentang efektivitas pembakaran kalori dalam berbagai aktivitas gym memiliki implikasi penting untuk pengembangan strategi intervensi kesehatan yang lebih efektif. Hal ini menjadi semakin relevan mengingat meningkatnya prevalensi penyakit terkait gaya hidup sedenter, seperti obesitas, diabetes, dan penyakit kardiovaskular. Optimalisasi program latihan berdasarkan pemahaman yang lebih baik tentang efektivitas pembakaran kalori dapat membantu mengatasi tantangan kesehatan ini dengan lebih efektif, sambil mempertimbangkan keterbatasan waktu dan sumber daya yang dihadapi masyarakat modern. Dengan demikian, penelitian tentang efektivitas pembakaran kalori dalam aktivitas gym tidak hanya berkontribusi pada pengembangan program kebugaran yang lebih efisien, tetapi juga berperan penting dalam upaya yang lebih luas untuk meningkatkan kesehatan masyarakat dan mengurangi beban penyakit kronis.

Penerapan machine learning memungkinkan identifikasi perilaku gym yang paling efektif, dengan algoritma seperti k-Nearest Neighbors (KNN), Random Forest, dan Boosting. KNN dapat menyarankan latihan berdasarkan karakteristik individu, sementara Random Forest menggabungkan berbagai faktor untuk prediksi pembakaran kalori yang akurat. Boosting meningkatkan ketepatan rekomendasi latihan dengan terus belajar dari kesalahan sebelumnya. Dengan model ini, pusat kebugaran bisa memberikan rekomendasi yang lebih dipersonalisasi, membantu pengguna mencapai tujuan kebugaran secara efisien sekaligus mendukung peningkatan kesehatan masyarakat melalui gaya hidup aktif. Data yang digunakan diambil dari kaggle yang bisa diakses dari link [berikut](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset/data)




Dataset ini mencakup profil kebugaran individu, meliputi detail demografis (usia, jenis kelamin), komposisi tubuh (berat badan, tinggi badan, BMI, persentase lemak), metrik detak jantung (Max_BPM, Avg_BPM, Resting_BPM), dan kebiasaan olahraga (Jenis Olahraga, Durasi Sesi, Kalori Terbakar, Frekuensi Olahraga). **BMI** dan **Persentase Lemak** memberikan gambaran tentang komposisi tubuh, dengan **Persentase Lemak** yang biasanya memberikan gambaran yang lebih akurat dibandingkan BMI, terutama untuk individu dengan massa otot yang tinggi. Metrik detak jantung menyoroti kebugaran kardiovaskular, di mana **Resting_BPM** sering kali lebih rendah pada individu yang lebih fit.

Data olahraga menunjukkan intensitas dan preferensi, dengan aktivitas berintensitas tinggi (seperti HIIT atau Kardio) yang cenderung membakar lebih banyak kalori dan memiliki **Avg_BPM** lebih tinggi dibandingkan dengan latihan berintensitas rendah seperti Yoga. **Asupan Air** dan **Tingkat Pengalaman** menambah kedalaman informasi, menunjukkan kebiasaan hidrasi dan tingkat keakraban dengan kebugaran, yang dapat memengaruhi hasil latihan dan detak jantung saat istirahat. Dataset ini memungkinkan pemahaman yang luas tentang tingkat kebugaran individu dan memberikan wawasan yang berguna untuk personalisasi rencana kebugaran dan kesehatan.


## Business Understanding
### Problem Statement
1. Bagaimana cara meningkatkan efektifitas jumlah Kalori Terbakar pada Latihan?
2. Bagaimana cara optimalkan latihan, jika ingin mencapai target Kadar Lemak Tubuh tertentu?
3. Apakah ada perbedaan antara Laki-laki dan Perempuan dalam preferensi latihan?
4. Apakah Tingkat Kemahiran mempengaruhi preferensi latihan?

### Goals
1. Membuat model yang memprediksi Kalori Terbakar dalam Latihan.
2. Membuat model yang memprediksi Kadar Lemak Tubuh.
3. Mencari perbedaan preferensi Laki-laki dan Perempuan.
4. Mencari preferensi setiap tingkat Kemahiran.

### Solution
1. Menggunakan 4 Algoritma Machine Learning untuk membuat model yang memprediksi Kalori Terbakar dan Kadar Lemak Tubuh.
2. Menggunakan Exploratory Data Analysis (EDA) untuk menentukan pengaruh Jenis Kelamin dan Tingkat Kemahiran dalam preferensi Latihan.


## **Pemahaman Data**


### Pemuatan Data

Data diambil dari kaggle pada tautan berikut https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset/data, dengan usability 10.00 dan view sebesar 103k pada saat diakses, yang selanjutnya diunduh ke dalam root sistem Google Colab. Data dipindahkan ke dalam drive agar dapat memudahkan penggunaan di Google Colab.
Data dibuka melalui dataframe dan ditampilkan sekilas menggunakan fungsi `.head()` dan `.shape` sebagai berikut

  
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Weight (kg)</th>
      <th>Height (m)</th>
      <th>Max_BPM</th>
      <th>Avg_BPM</th>
      <th>Resting_BPM</th>
      <th>Session_Duration (hours)</th>
      <th>Calories_Burned</th>
      <th>Workout_Type</th>
      <th>Fat_Percentage</th>
      <th>Water_Intake (liters)</th>
      <th>Workout_Frequency (days/week)</th>
      <th>Experience_Level</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56</td>
      <td>Male</td>
      <td>88.3</td>
      <td>1.71</td>
      <td>180</td>
      <td>157</td>
      <td>60</td>
      <td>1.69</td>
      <td>1313.0</td>
      <td>Yoga</td>
      <td>12.6</td>
      <td>3.5</td>
      <td>4</td>
      <td>3</td>
      <td>30.20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46</td>
      <td>Female</td>
      <td>74.9</td>
      <td>1.53</td>
      <td>179</td>
      <td>151</td>
      <td>66</td>
      <td>1.30</td>
      <td>883.0</td>
      <td>HIIT</td>
      <td>33.9</td>
      <td>2.1</td>
      <td>4</td>
      <td>2</td>
      <td>32.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>Female</td>
      <td>68.1</td>
      <td>1.66</td>
      <td>167</td>
      <td>122</td>
      <td>54</td>
      <td>1.11</td>
      <td>677.0</td>
      <td>Cardio</td>
      <td>33.4</td>
      <td>2.3</td>
      <td>4</td>
      <td>2</td>
      <td>24.71</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>Male</td>
      <td>53.2</td>
      <td>1.70</td>
      <td>190</td>
      <td>164</td>
      <td>56</td>
      <td>0.59</td>
      <td>532.0</td>
      <td>Strength</td>
      <td>28.8</td>
      <td>2.1</td>
      <td>3</td>
      <td>1</td>
      <td>18.41</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38</td>
      <td>Male</td>
      <td>46.1</td>
      <td>1.79</td>
      <td>188</td>
      <td>158</td>
      <td>68</td>
      <td>0.64</td>
      <td>556.0</td>
      <td>Strength</td>
      <td>29.2</td>
      <td>2.8</td>
      <td>3</td>
      <td>1</td>
      <td>14.39</td>
    </tr>
  </tbody>
</table>

Tampilan diatas menunjukkan ada 973 rekod data dengan 15 jenis informasi yang dapat dianalisis.

### Analisis Data Eksploratif (*EDA*)


#### Deskripsi Variabel

##### Arti Variabel

Berikut ini adalah arti dari setiap variabel yang akan digunakan.

| Variabel                           | Keterangan |
|-------------------------------------|-------------------|
| Age                                  | Umur dari pengunjung pusat kebugaran dalam Tahun|
| Gender                               |Jenis kelamin dari pengunjung pusat kebugaran (Pria, Wanita) |
| Weight (kg)                          | Berat badan pengunjung pusat kebugaran dalam Kilogram |
| Height (m)                           |Tinggi pengunjung pusat kebugaran dalam Centimeter|
| Max_BPM                              | Laju maksimal detak jantung permenit peserta|
| Avg_BPM                              |Laju rata-rata detak jantung permenit peserta  |
| Resting_BPM                          | Laju detak jantung istirahat permenit peserta|
| Session_Duration (hours)            | Durasi penggunaan pusat kebugaran dalam Jam|
| Calories_Burned                      | Jumlah kalori yang terbakar dalam Kal|
| Workout_Type                         | Jenis Latihan yang dilakukan pelanggan (HIIT, Strength, Cardio, Yoga)|
| Fat_Percentage                       | Kadar lemak tubuh dari pengunjung dalam persen (%) |
| Water_Intake (liters)               | Jumlah air yang diminum selama latihan dalam Liter |
| Workout_Frequency (days/week)       | Frekuensi kunjungan ke pusat kebugaran dalam seminggu |
| Experience_Level                     | Tingkat kemahiran dalam melakukan latihan dalam tiga tingkatan (1, 2, 3) |
| BMI                                  | Indeks massa tubuh dari pengunjung pusat kebugaran                 |


Beberapa variabel merupakan variabel kategorik, dan sisanya adalah variabel numerik. Penjelasan lebih lanjut mengenai nilai variabel sebagai berikut.

Variabel `Workout_Type` yang berarti Jenis Latihan memiliki penjelasan nilai variabel seperti berikut

| Variabel                           | Keterangan |
|-------------------------------------|-------------------|
| HIIT| *High-Intensity Interval Training* yaitu latihan intensitas tinggi yang dilakukan dengan durasi cepat disertai istirahat dan pergantian gerakan|
| Strength| Olahraga latihan untuk meningkatkan kekuatan fisik seperti angkat beban atau *bench-press* |
| Yoga| Olahraga yang menggabungkan gerakan, pernapasan, dan meditasi untuk meningkatkan kesehatan fisik dan keseimbangan mental.|
|Cardio|Olahraga yang meningkatkan laju detak jantung dan laju nafas dengan berbagai intensitas seperti bersepeda, lari, olahraga *treadmill*|



##### Tipe Variabel

```
  <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 973 entries, 0 to 972
    Data columns (total 17 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   Age                            973 non-null    int64  
     1   Gender                         973 non-null    object 
     2   Weight (kg)                    973 non-null    float64
     3   Height (m)                     973 non-null    float64
     4   Max_BPM                        973 non-null    int64  
     5   Avg_BPM                        973 non-null    int64  
     6   Resting_BPM                    973 non-null    int64  
     7   Session_Duration (hours)       973 non-null    float64
     8   Calories_Burned                973 non-null    float64
     9   Workout_Type                   973 non-null    object 
     10  Fat_Percentage                 973 non-null    float64
     11  Water_Intake (liters)          973 non-null    float64
     12  Workout_Frequency (days/week)  973 non-null    int64  
     13  Experience_Level               973 non-null    int64  
     14  BMI                            973 non-null    float64
     15  Workout_Frequency_cat          973 non-null    object 
     16  Experience_Level_cat           973 non-null    object 
    dtypes: float64(7), int64(6), object(4)
    memory usage: 129.4+ KB
```

Ditemukan ada enam variabel bertipe int64, dua variabel bertipe object dan tujuh variabel bertipe float64. Selanjutnya dapat dilihat bahwa seluruh data bertipe float64 adalah variabel numerik dan seluruh data bertipe object adalah variabel kategorik. Sedangkan data bertipe int64, dua diantaranya bisa digunakan sebagai variabel kategorik dan empat diantaranya adalah variabel numerik.

##### Deskripsi statistik dari data

 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Weight (kg)</th>
      <th>Height (m)</th>
      <th>Max_BPM</th>
      <th>Avg_BPM</th>
      <th>Resting_BPM</th>
      <th>Session_Duration (hours)</th>
      <th>Calories_Burned</th>
      <th>Fat_Percentage</th>
      <th>Water_Intake (liters)</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>973.000000</td>
      <td>973.000000</td>
      <td>973.00000</td>
      <td>973.000000</td>
      <td>973.000000</td>
      <td>973.000000</td>
      <td>973.000000</td>
      <td>973.000000</td>
      <td>973.000000</td>
      <td>973.000000</td>
      <td>973.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.683453</td>
      <td>73.854676</td>
      <td>1.72258</td>
      <td>179.883864</td>
      <td>143.766701</td>
      <td>62.223022</td>
      <td>1.256423</td>
      <td>905.422405</td>
      <td>24.976773</td>
      <td>2.626619</td>
      <td>24.912127</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.180928</td>
      <td>21.207500</td>
      <td>0.12772</td>
      <td>11.525686</td>
      <td>14.345101</td>
      <td>7.327060</td>
      <td>0.343033</td>
      <td>272.641516</td>
      <td>6.259419</td>
      <td>0.600172</td>
      <td>6.660879</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>40.000000</td>
      <td>1.50000</td>
      <td>160.000000</td>
      <td>120.000000</td>
      <td>50.000000</td>
      <td>0.500000</td>
      <td>303.000000</td>
      <td>10.000000</td>
      <td>1.500000</td>
      <td>12.320000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
      <td>58.100000</td>
      <td>1.62000</td>
      <td>170.000000</td>
      <td>131.000000</td>
      <td>56.000000</td>
      <td>1.040000</td>
      <td>720.000000</td>
      <td>21.300000</td>
      <td>2.200000</td>
      <td>20.110000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>40.000000</td>
      <td>70.000000</td>
      <td>1.71000</td>
      <td>180.000000</td>
      <td>143.000000</td>
      <td>62.000000</td>
      <td>1.260000</td>
      <td>893.000000</td>
      <td>26.200000</td>
      <td>2.600000</td>
      <td>24.160000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49.000000</td>
      <td>86.000000</td>
      <td>1.80000</td>
      <td>190.000000</td>
      <td>156.000000</td>
      <td>68.000000</td>
      <td>1.460000</td>
      <td>1076.000000</td>
      <td>29.300000</td>
      <td>3.100000</td>
      <td>28.560000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>59.000000</td>
      <td>129.900000</td>
      <td>2.00000</td>
      <td>199.000000</td>
      <td>169.000000</td>
      <td>74.000000</td>
      <td>2.000000</td>
      <td>1783.000000</td>
      <td>35.000000</td>
      <td>3.700000</td>
      <td>49.840000</td>
    </tr>
  </tbody>
</table>



Informasi statistik tersebut menunjukkan simpangan baku yang cukup tinggi yang menandakan perbedaan profil yang cukup signifikan antar responden. Beragam profil dalam data bisa berarti adanya berbagai demografi yang tercangkup dari data tersebut yang dapat dianalisis.

##### Pembersihan Data


 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nilai yang Kosong</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Weight (kg)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Height (m)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Max_BPM</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Avg_BPM</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Resting_BPM</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Session_Duration (hours)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Calories_Burned</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Workout_Type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Fat_Percentage</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Water_Intake (liters)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Workout_Frequency (days/week)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Experience_Level</th>
      <td>0</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>0</td>
    </tr>
  </tbody>
</table>





Tidak ditemukan adanya data yang kosong dan data ganda sehingga tidak perlu dilakukan pembersihan data lebih lanjut.

##### Pencilan

*Boxplot* digunakan untuk melihat penyebaran data.


    
![png](gambar_files/gambar_32_0.png)
    


Tampilan diatas menunjukkan tiga kolom informasi yang mempunyai nilai datum diluar batas kuartil. Kolom berat badan dan BMI mempunyai beberapa potensi pencilan yaitu beberapa datum yang berada diatas nilai kuartil atas. Namun data ini masih terbilang wajar mengingat pusat kebugaran merupakan tempat yang mengakomodasi program penurunan berat badan jadi memiliki berat badan tinggi bukan merupakan anomali. Data kalori terbakar juga memiliki beberapa datum yang berada diluar kuartil atas. Ini bukan merupakan pencilan yang harus dihilangkan karena sangat memungkinkan untuk seseorang mahir dalam latihan kebugaran sehingga dapat membakar kalori lebih banyak dalam satu waktu.

##### Koreksi tipe data

Dua kolom informasi bertipe int64 dapat diperlakukan sebagai data kategorik. Oleh karena itu dibuatlah kolom baru mengunakan data tersebut dengan mengubah jenis data menjadi string. Selanjutnya string tersebut diubah menjadi tipe data object agar bisa dikenali filter tipe object seperti data kategorik lainnya.


#### Analisis Univariat

Data dipisahkan anatara variabel numerik dan kategorik sebagai berikut.

    Kolom-kolom numerik:  ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']
    Kolom-kolom kategorik:  ['Gender', 'Workout_Type', 'Workout_Frequency_cat', 'Experience_Level_cat']

 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Nilai Berbeda</th>
      <th>Nilai-Nilai</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gender</th>
      <td>2</td>
      <td>[Male, Female]</td>
    </tr>
    <tr>
      <th>Workout_Type</th>
      <td>4</td>
      <td>[Yoga, HIIT, Cardio, Strength]</td>
    </tr>
    <tr>
      <th>Workout_Frequency_cat</th>
      <td>4</td>
      <td>[4, 3, 5, 2]</td>
    </tr>
    <tr>
      <th>Experience_Level_cat</th>
      <td>3</td>
      <td>[3, 2, 1]</td>
    </tr>
  </tbody>
</table>

Ada 4 variabel kategorik yang bisa digunakan untuk mengelompokkan data.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>Workout_Type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Strength</th>
      <td>258</td>
    </tr>
    <tr>
      <th>Cardio</th>
      <td>255</td>
    </tr>
    <tr>
      <th>Yoga</th>
      <td>239</td>
    </tr>
    <tr>
      <th>HIIT</th>
      <td>221</td>
    </tr>
  </tbody>
</table>


    
![png](gambar_files/gambar_42_0.png)
    

Distribusi jumlah pelanggan sesuai jenis latihan yang dilakukan cukup tersebar merata dengan latihan Strength sebagai latihan yang paling banyak dilakukan meskipun tidak jauh berbeda dengan jenis latihan lainnya.

    
![png](gambar_files/gambar_44_0.png)
    


Pengunjung pusat kebugaran lebih banyak laki-laki meskipun bedanya tidak banyak dengan pengunjung perempuan. Pengunjung yang paling banyak datang ke pusat kebugaran adalah yang datang dua kali seminggu. Jumlah anggota variabel `Tingkat Kemahiran` cukup timpang dengan `Tingkat Kemahiran 3` mempunyai anggota paling sedikit dibanding yang lain.





    
![png](gambar_files/gambar_46_0.png)
    


Variabel `Durasi Latihan` dan `Kalori Terbakar` memiliki data yang terdistribusi cukup normal. Variabel `Berat Badan`,`Tinggi Badan`, dan `BMI` memiliki distribusi yang miring ke kanan yang berarti kebanyakan anggotanya berada di bawah rata-rata. Sedangkan variabel `Kadar Lemak Tubuh` dan `Asupan Air` memiliki distribusi yang miring ke kiri yaitu anggotanya lebih banyak yang memiliki nilai diatas rata-rata. Lalu sisanya yaitu `Usia`, `Max BPM`, `Resting BPM`, `Avg BPM`, `Frekuensi Latihan` dan `Tingkat Kemahiran` tidak berdistribusi normal.



#### Analisis Multivariat

##### 1. Analisis Berbagai Distribusi Kategori Berdasarkan Jenis Kelamin





    
![png](gambar_files/gambar_50_0.png)
    


Distribusi penyebaran data antara laki-laki dan perempuan memiliki penyebaran yang hampir sama pada setiap kategori. Perbedaan hanya terlihat pada kategori `Jenis Latihan` yaitu perempuan lebih terdistribusi ke `Kardio` sedangkan laki-laki surplus distribusi pada jenis latihan `Yoga`.

##### 2. Distribusi Jenis Latihan Berdasarkan Level Kemahiran





    
![png](gambar_files/gambar_53_0.png)
    


`Tingkat Kemahiran` tidak memengaruhi preferensi jenis latihan yang dilakukan karena secara persentase distribusinya serupa. Namun terlihat dari jumlah total `Tingkat Kemahiran` 1 dan 2 mayoritas melakukan latihan `Strength` sedangkan pada `Tingkat Kemahiran` 3 lebih banyak yang melakukan `Yoga`.

##### 3. Distribusi Nilai Kadar Lemak Tubuh dari berbagai Kategori







    
![png](gambar_files/gambar_56_0.png)
    


`Tingkat Kemahiran` 3 mempunyai `Kadar Lemak Tubuh` antara 10-15% sedangkan Tingkat Kemahiran 1 dan 2 antara 20-35%. `Kadar Lemak Tubuh` yang relatif tinggi ditemukan pada pengunjung yang datang dua sampai tiga kali seminggu yaitu antara 20-35%. Pengunjung yang datang lima kali dalam seminggu mempunyai rentang `Kadar Lemak Tubuh` relatif rendah antar 10-20% sedangkan pada kelompok pengunjung yang datang empat kali dalam seminggu tersebar antara 10-35% `Kadar Lemak Tubuh`.

##### 4. Distribusi Kalori Terbakar dari berbagai Kategori





    
![png](gambar_files/gambar_59_0.png)
    


Tingkat Kemahiran 1 memiliki sebaran luas, 400-1300 kalori terbakar per latihan. Tingkat Kemahiran 2 lebih terfokus, 600-1400 kalori. Tingkat Kemahiran 3 memiliki sebaran jarang dengan rentang tinggi, 900-1800 kalori.
Orang yang datang 2-3 kali seminggu memiliki sebaran 400-1400 kalori terbakar per latihan. Pengunjung dengan kehadiran 4 kali seminggu lebih tersebar, 600-1800 kalori. Pengunjung yang datang 5 kali seminggu memiliki nilai rentang tinggi, 900-1800 kalori.

##### 5. Distribusi Indeks Massa Tubuh dari berbagai Kategori





    
![png](gambar_files/gambar_62_0.png)
    



Tingkat kemahiran 1, nilai IMT tersebar luas dari sekitar 15 hingga 50. Pada tingkat kemahiran 2, distribusi menjadi lebih sempit dengan lebih sedikit pencilan pada IMT yang tinggi. Sementara itu, pada tingkat kemahiran 3, nilai IMT lebih terkonsentrasi dalam rentang 25-35.

Peserta yang berlatih 2-3 kali seminggu memiliki nilai IMT yang tersebar luas, termasuk beberapa pencilan dengan IMT di atas 40. Namun, ketika frekuensi latihan meningkat menjadi 4-5 kali seminggu, distribusi IMT lebih terkonsentrasi pada rentang 20-35.

Peserta Yoga dan HIIT memiliki distribusi IMT yang cukup luas, termasuk beberapa dengan IMT di atas 40. Sebaliknya, peserta yang memilih latihan Kardio dan Latihan Kekuatan cenderung lebih terkonsentrasi dalam rentang IMT 25-35.

##### 6. *Heat Map*





    
![png](gambar_files/gambar_65_0.png)
    


Berdasarkan heatmap korelasi, BMI memiliki korelasi positif yang sangat kuat dengan berat badan (0.85), karena memang merupakan faktor utama dalam perhitungan BMI. Frekuensi latihan berhubungan positif dengan durasi sesi latihan (0.64) dan pembakaran kalori (0.36), yang juga mencerminkan bahwa tingkat kemahiran meningkat seiring dengan lebih banyaknya frekuensi dan durasi latihan (korelasi dengan tingkat kemahiran adalah 0.69 dan 0.76). Selain itu, asupan air memiliki hubungan positif dengan frekuensi latihan (0.44) dan durasi latihan (0.28), menunjukkan bahwa peserta yang lebih aktif cenderung mengonsumsi lebih banyak air. Di sisi lain, persentase lemak tubuh memiliki korelasi negatif dengan pembakaran kalori (-0.60) dan frekuensi latihan (-0.54), mengindikasikan bahwa aktivitas fisik yang lebih sering dan pembakaran kalori yang lebih tinggi cenderung berhubungan dengan lemak tubuh yang lebih rendah. Secara keseluruhan, aktivitas fisik yang teratur dan intens berkontribusi pada tingkat kebugaran yang lebih baik, asupan air yang lebih tinggi, dan lemak tubuh yang lebih rendah.

##### 7. *Pair Plot




    
![png](gambar_files/gambar_68_1.png)
    


Berdasarkan pairplot, variabel numerik seperti berat badan, tinggi badan, dan durasi sesi latihan menunjukkan distribusi yang mendekati normal, sementara variabel seperti frekuensi latihan dan tingkat kemahiran bersifat kategorikal. Hubungan positif yang kuat terlihat antara BMI dan berat badan, serta antara kalori terbakar dan durasi sesi latihan, menunjukkan bahwa berat badan memengaruhi BMI dan durasi sesi yang lebih lama menghasilkan pembakaran kalori yang lebih banyak. Selain itu, asupan air cenderung meningkat seiring dengan frekuensi latihan yang lebih tinggi. Di sisi lain, variabel seperti detak jantung (Max_BPM dan Resting_BPM) tidak menunjukkan hubungan yang jelas dengan variabel lain. Beberapa pencilan teridentifikasi pada variabel seperti BMI dan kalori terbakar, yang dapat mencerminkan pola unik pada peserta tertentu. Secara keseluruhan, pairplot ini menunjukkan beberapa hubungan signifikan antarvariabel, meskipun beberapa variabel lain memiliki korelasi yang lemah atau tidak jelas, mencerminkan keragaman data peserta.

##### 8. Perbandingan Kalori Terbakar dengan Kadar Lemak Tubuh




    
![png](gambar_files/gambar_71_0.png)
    


Terdapat korelasi negatif antara kedua variabel antara Kalori Terbakar saat latihan dengan Kadar Lemak Tubuh. Semakin banyak seseorang memiliki kalori terbakar saat latihan mengindikasikan bahwa orang tersebut Kadar Lemak Tubuhnya semakin rendah.

##### 9. Perbandingan Antara Kadar Lemak Tubuh dengan Berbagai Variabel Numerik





    
![png](gambar_files/gambar_74_0.png)
    


Grafik ini menunjukkan hubungan antara persentase lemak tubuh dengan beberapa variabel numerik lainnya. Pada hubungan antara **persentase lemak tubuh dan asupan air (Water Intake)** terlihat adanya korelasi negatif, di mana semakin tinggi persentase lemak tubuh, rata-rata asupan air cenderung menurun. Pola serupa juga terlihat pada hubungan dengan **durasi latihan (Session Duration)**, yang menunjukkan bahwa individu dengan persentase lemak tubuh lebih tinggi cenderung memiliki durasi olahraga yang lebih pendek. Namun, untuk hubungan dengan variabel seperti **denyut jantung maksimum (Max BPM)**, **denyut jantung rata-rata (Avg BPM)**, dan **denyut jantung istirahat (Resting BPM)**, tidak ditemukan korelasi yang signifikan, karena data tampak tersebar secara acak. Begitu pula untuk **usia**, grafik menunjukkan bahwa persentase lemak tubuh tidak memiliki hubungan yang jelas dengan umur. Secara keseluruhan, hubungan paling menonjol adalah antara lemak tubuh dengan asupan air dan durasi olahraga, sedangkan variabel lain tidak menunjukkan pola yang jelas.

##### 10. Perbandingan Antara Kalori Terbakar dengan Berbagai Variabel Numerik





    
![png](gambar_files/gambar_77_0.png)
    


Grafik ini menunjukkan hubungan antara jumlah kalori yang terbakar dengan berbagai variabel numerik lainnya. Hubungan yang paling kuat terlihat pada grafik **Calories Burned vs. Session Duration**, di mana semakin lama durasi sesi olahraga, semakin banyak kalori yang terbakar, menunjukkan korelasi linear yang jelas. Pada hubungan dengan **water intake**, terdapat korelasi positif ringan, di mana asupan air cenderung meningkat seiring dengan kalori yang terbakar, meskipun data tersebar cukup luas. Korelasi positif juga terlihat pada hubungan dengan **Average BPM (denyut jantung rata-rata)**, di mana kalori yang terbakar cenderung lebih tinggi pada aktivitas dengan denyut rata-rata lebih besar. Sebaliknya, variabel seperti **Max BPM (denyut jantung maksimum)** dan **Resting BPM (denyut istirahat)** tidak menunjukkan pola hubungan yang signifikan, dengan data yang tersebar acak. Pada hubungan dengan **usia**, terdapat pola negatif lemah, di mana jumlah kalori yang terbakar sedikit menurun pada individu yang lebih tua, meskipun hubungan ini tidak terlalu signifikan. Secara keseluruhan, durasi sesi olahraga memiliki hubungan paling signifikan dengan jumlah kalori yang terbakar, sementara variabel lainnya menunjukkan hubungan yang lemah atau tidak signifikan.


## **Persiapan Data**

Memanfaatkan beberapa fitur baru yang bisa dibuat dari beberapa variabel yang ada agar didapat akurasi yang lebih maksimal dalam model maka akan dilakukan rekayasa fitur.

### **Rekayasa Fitur**

**Intensity Score**

Rekayasa fitur untuk menemukan **Intensity Score**, sebuah metrik yang bertujuan untuk merepresentasikan tingkat intensitas aktivitas fisik. Intensity Score dihitung dengan menggabungkan beberapa variabel terkait, seperti durasi sesi olahraga, denyut jantung rata-rata (Avg BPM), denyut jantung maksimum (Max BPM), dan kalori yang terbakar, menggunakan pendekatan berbobot atau transformasi matematis tertentu. Fitur ini diharapkan mampu memberikan gambaran yang lebih holistik tentang tingkat usaha yang dilakukan seseorang selama aktivitas fisik, sehingga dapat meningkatkan akurasi analisis dan prediksi dalam model yang digunakan.



    
![png](gambar_files/gambar_81_0.png)
    


`Skor intensitas` mempunyai hubungan signifikan dengan `Kadar Lemak Tubuh` dan jumlah `Kalori Terbakar`. Semakin tinggi `skor intensitas` maka biasanya orang tersebut mempunyai kadar lemak tubuh yang lebih rendah. Sedangkan pada `kalori terbakar` semakin tinggi intensitas maka semakin tinggi pula `kalori terbakar`.

**HR Index**  

Langkah ini melakukan rekayasa fitur untuk menghasilkan `HR Index`, sebuah metrik yang dirancang untuk merepresentasikan tingkat intensitas aktivitas fisik berdasarkan variabel-variabel terkait denyut jantung. Fitur ini dihitung dengan menggabungkan `HR Index (Heart Rate Index)`, yang mencakup rasio antara denyut jantung saat aktivitas dan denyut jantung istirahat. HR Index diharapkan dapat memberikan gambaran yang lebih komprehensif mengenai tingkat usaha individu selama aktivitas fisik, sehingga dapat digunakan sebagai indikator utama dalam analisis performa atau prediksi tingkat kebugaran.





    
![png](gambar_files/gambar_84_0.png)
    


Grafik ini menunjukkan hubungan antara `HR Index (Heart Rate Index)` dengan `Fat Percentage` dan `Calories Burned`. Hubungan antara `Fat Percentage` dan `HR Index`, terlihat bahwa tidak ada korelasi signifikan, dengan data yang tersebar acak di sekitar garis regresi yang mendatar, menunjukkan bahwa HR Index tidak dipengaruhi oleh persentase lemak tubuh. Sementara itu, hubungan antara `Calories Burned` dan `HR Index` menunjukkan korelasi negatif yang lemah, di mana `HR Index` sedikit menurun seiring meningkatnya jumlah kalori yang terbakar, meskipun hubungan ini tidak terlalu kuat karena data masih tersebar di sekitar garis regresi.

### **Reduksi Variabel**

Kolom `Workout_Frequency_cat` dan `Experiences_Level_cat` sejatinya adalah kolom yang redundan karena sudah ada dalam data sebagai numerikal dalam bentuk integer atau bilangan bulat.  Tidak diperlukan encoding lebih lanjut karena sudah setara posisinya dengan bilangan ordinal.

### **Encoding**

Kolom `Gender` dan `Workout_Type` dilakukan encoding dengan menggunakan `LabelEncoder()` yang menghasilkan output encoding ordinal.

### **Normalisasi Data**

Data yang sudah dalam bentuk numerik dari beberapa proses diatas akan dinormalisasi untuk mengubah nilai rata-rata setiap variabel menjadi 0, nilai maksimum sebesar 1, dan nilai minimum sebesar -1. Hal ini dilakukan agar pemodelan menjadi lebih konsisten karena memiliki batas atas dan bawah yang seragam. Proses ini menggunakan `StandarScaler` untuk semua nilai numerik.

### **Train-Test-Split**

Data kemudian displit dengan perbandingan 80% training data dan 20% test data, proses ini dilakukan secara acak dengan `random state` bernilai 42.

#### Model Kalori Terbakar

Pembuatan data untuk model Kalori Terbakar yaitu dengan mendrop kolom Kalori Terbakar dari keseluruhan Data untuk membentuk nilai X sebagai variabel bebas. Semua kolom digunakan kecuali kolom Kalori Terbakar yang akan digunakan sebagai y atau variabel terikat.


 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Weight (kg)</th>
      <th>Height (m)</th>
      <th>Max_BPM</th>
      <th>Avg_BPM</th>
      <th>Resting_BPM</th>
      <th>Session_Duration (hours)</th>
      <th>Workout_Type</th>
      <th>Fat_Percentage</th>
      <th>Water_Intake (liters)</th>
      <th>Workout_Frequency (days/week)</th>
      <th>Experience_Level</th>
      <th>BMI</th>
      <th>Intensity_Score</th>
      <th>HR_Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>967</th>
      <td>-1.534617</td>
      <td>0.950847</td>
      <td>-0.889514</td>
      <td>-0.960247</td>
      <td>-0.684377</td>
      <td>1.690177</td>
      <td>0.652299</td>
      <td>-0.397902</td>
      <td>1.338485</td>
      <td>-0.156129</td>
      <td>0.955853</td>
      <td>0.743295</td>
      <td>0.257176</td>
      <td>-0.515531</td>
      <td>-1.016628</td>
      <td>-1.563369</td>
    </tr>
    <tr>
      <th>365</th>
      <td>0.026000</td>
      <td>-1.051694</td>
      <td>-0.625321</td>
      <td>-0.568564</td>
      <td>-1.552451</td>
      <td>1.620431</td>
      <td>0.106097</td>
      <td>-0.981235</td>
      <td>-0.439462</td>
      <td>0.419300</td>
      <td>-1.544716</td>
      <td>-0.352502</td>
      <td>-1.095432</td>
      <td>-0.398370</td>
      <td>-1.479334</td>
      <td>-1.812168</td>
    </tr>
    <tr>
      <th>559</th>
      <td>1.258067</td>
      <td>-1.051694</td>
      <td>0.082340</td>
      <td>-0.020208</td>
      <td>1.225384</td>
      <td>0.713732</td>
      <td>-0.303555</td>
      <td>-0.835402</td>
      <td>-1.328435</td>
      <td>0.227491</td>
      <td>-1.211307</td>
      <td>-0.352502</td>
      <td>-1.095432</td>
      <td>0.095813</td>
      <td>-0.585599</td>
      <td>0.331517</td>
    </tr>
    <tr>
      <th>33</th>
      <td>-1.206066</td>
      <td>-1.051694</td>
      <td>-0.705522</td>
      <td>-1.665277</td>
      <td>0.617733</td>
      <td>0.922970</td>
      <td>0.788850</td>
      <td>-0.631235</td>
      <td>-1.328435</td>
      <td>1.074650</td>
      <td>-0.211079</td>
      <td>-1.448299</td>
      <td>-1.095432</td>
      <td>0.137871</td>
      <td>-0.752430</td>
      <td>-0.535557</td>
    </tr>
    <tr>
      <th>31</th>
      <td>-1.534617</td>
      <td>-1.051694</td>
      <td>-0.398869</td>
      <td>-1.586940</td>
      <td>0.444118</td>
      <td>-1.169411</td>
      <td>-1.669061</td>
      <td>-0.660402</td>
      <td>1.338485</td>
      <td>0.483237</td>
      <td>-0.711193</td>
      <td>0.743295</td>
      <td>0.257176</td>
      <td>0.510386</td>
      <td>0.165142</td>
      <td>1.822367</td>
    </tr>
  </tbody>
</table>

### **Model Kadar Lemak Tubuh**

Proses pembuatan data untuk model Kadar Lemak Tubuh dilakukan dengan menghapus kolom Kadar Lemak Tubuh dari seluruh dataset, sehingga menghasilkan nilai X sebagai variabel bebas. Semua kolom lainnya digunakan, kecuali kolom Kadar Lemak Tubuh yang akan menjadi variabel terikat atau y.

 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Weight (kg)</th>
      <th>Height (m)</th>
      <th>Max_BPM</th>
      <th>Avg_BPM</th>
      <th>Resting_BPM</th>
      <th>Session_Duration (hours)</th>
      <th>Calories_Burned</th>
      <th>Workout_Type</th>
      <th>Water_Intake (liters)</th>
      <th>Workout_Frequency (days/week)</th>
      <th>Experience_Level</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.422343</td>
      <td>0.950847</td>
      <td>0.681493</td>
      <td>-0.098545</td>
      <td>0.010081</td>
      <td>0.922970</td>
      <td>-0.303555</td>
      <td>1.264598</td>
      <td>1.495690</td>
      <td>1.338485</td>
      <td>1.455967</td>
      <td>0.743295</td>
      <td>1.609784</td>
      <td>0.794278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.600965</td>
      <td>-1.051694</td>
      <td>0.049316</td>
      <td>-1.508604</td>
      <td>-0.076726</td>
      <td>0.504494</td>
      <td>0.515749</td>
      <td>0.127098</td>
      <td>-0.082284</td>
      <td>-0.439462</td>
      <td>-0.877898</td>
      <td>0.743295</td>
      <td>0.257176</td>
      <td>1.064652</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.548964</td>
      <td>-1.051694</td>
      <td>-0.271491</td>
      <td>-0.490228</td>
      <td>-1.118414</td>
      <td>-1.518142</td>
      <td>-1.122858</td>
      <td>-0.427068</td>
      <td>-0.838243</td>
      <td>-1.328435</td>
      <td>-0.544488</td>
      <td>0.743295</td>
      <td>0.257176</td>
      <td>-0.030361</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.123928</td>
      <td>0.950847</td>
      <td>-0.974433</td>
      <td>-0.176881</td>
      <td>0.878155</td>
      <td>1.411193</td>
      <td>-0.849757</td>
      <td>-1.943735</td>
      <td>-1.370351</td>
      <td>0.449512</td>
      <td>-0.877898</td>
      <td>-0.352502</td>
      <td>-1.095432</td>
      <td>-0.976669</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.056137</td>
      <td>0.950847</td>
      <td>-1.309393</td>
      <td>0.528148</td>
      <td>0.704540</td>
      <td>0.992716</td>
      <td>0.788850</td>
      <td>-1.797902</td>
      <td>-1.282278</td>
      <td>0.449512</td>
      <td>0.289035</td>
      <td>-0.352502</td>
      <td>-1.095432</td>
      <td>-1.580503</td>
    </tr>
  </tbody>
</table>

## **Pemodelan**

Model yang akan dibuat ada dua yaitu `Model Kalori Terbakar` yang digunakan untuk mencari faktor apa saja yang memeangaruhi efektifitas latihan dalam gym serta `Model Kadar Lemak Tubuh` yang digunakan untuk mencari kebiasaan-kebiasaan latihan apa saja yang harus dilakukan untuk mencapai target tipe tubuh tertentu.

Ada 4 algoritma yang dipilih dengan framework yang berbeda yaitu Random Forest, K-Nearest Neighbors, Support Vector Reressor dan XGboost. Random Forest (RF) unggul dalam menangani data kompleks dan besar, serta dapat mengurangi risiko overfitting dengan menggunakan banyak pohon keputusan tanpa membutuhkan praproses data yang rumit. Namun, proses pelatihan dan prediksi bisa lebih lambat dan sulit diinterpretasi. K-Nearest Neighbors (KNN) sederhana dan mudah dipahami, tidak memerlukan pelatihan, serta dapat digunakan untuk regresi dan klasifikasi, tetapi kinerjanya menurun pada dataset besar dan sangat sensitif terhadap noise serta data yang tidak seimbang. Support Vector Regression (SVR) efektif untuk data dengan dimensi tinggi dan linearitas kompleks serta dapat menangani noise, namun kurang efisien untuk dataset besar dan sulit diinterpretasi karena bergantung pada pemilihan kernel dan parameter yang tepat. XGBoost cepat, efisien, dan sering menghasilkan akurasi tinggi dengan regularisasi yang baik untuk mengurangi overfitting, namun dapat overfit jika tidak dikonfigurasi dengan benar, memerlukan pemilihan hyperparameter yang tepat, dan model yang dihasilkan sulit diinterpretasi.

### Random Forest
Random Forest (RF) adalah algoritma ensemble learning yang bekerja dengan membangun banyak pohon keputusan secara independen menggunakan subset data dan fitur yang dipilih secara acak. Prediksi akhir ditentukan melalui rata-rata (untuk regresi) atau voting mayoritas (untuk klasifikasi). RF unggul dalam menangani data kompleks dan besar serta dapat mengurangi risiko overfitting melalui kombinasi banyak pohon keputusan. Selain itu, algoritma ini tidak memerlukan praproses data yang rumit, sehingga dapat langsung digunakan pada data mentah dengan outlier atau nilai kosong. Namun, proses pelatihannya relatif lambat karena memerlukan waktu untuk membangun banyak pohon, dan hasilnya sulit diinterpretasi karena merupakan agregasi dari model-model individual.

Kedua model yang dibuat adalah Random Forest Regressor dengan n_estimators sebesar 100 dan random_state sebesar 42.

<table style="width: 100%; text-align: center;">
  <tbody>
    <tr>
      <td style="background-color: #add8e6; color: #000;">RandomForestRegressor</td>
    </tr>
    <tr>
      <td style="background-color: #e6f7ff; color: #000;"><code>RandomForestRegressor(random_state=42)</code></td>
    </tr>
  </tbody>
</table>

### K-Nearest Neighbors

K-Nearest Neighbors (KNN) adalah algoritma yang sederhana dan intuitif karena tidak memerlukan proses pelatihan. Algoritma ini bekerja dengan mencari sejumlah tetangga terdekat (k) dari titik data yang akan diprediksi menggunakan metrik jarak, seperti Euclidean, kemudian membuat prediksi berdasarkan rata-rata (untuk regresi) atau voting (untuk klasifikasi) dari tetangga tersebut. KNN sangat cocok untuk dataset kecil dan mudah dipahami, serta fleksibel untuk berbagai tipe distribusi data. Namun, algoritma ini memiliki kelemahan berupa kinerja yang lambat pada dataset besar karena perhitungan jarak yang mahal, serta sangat sensitif terhadap noise dan data yang tidak seimbang. 

Kedua model K-Nearest Neighbors yang dibuat adalah dengan jumlah n_neighbors=10.


<table style="width: 100%; text-align: center;">
  <tbody>
    <tr>
      <td style="background-color: #add8e6; color: #000;">KNeighborsRegressor</td>
    </tr>
    <tr>
      <td style="background-color: #e6f7ff; color: #000;"><code>KNeighborsRegressor(n_neighbors=10)</code></td>
    </tr>
  </tbody>
</table>






### Support Vector Regressor
Support Vector Regressor (SVR) adalah versi regresi dari Support Vector Machine (SVM) yang bekerja dengan mencari hyperplane terbaik dalam ruang dimensi tinggi untuk memprediksi nilai target. Algoritma ini menggunakan kernel seperti linear, polynomial, atau RBF untuk menangani hubungan data yang kompleks dan non-linear. SVR sangat efektif untuk data berdimensi tinggi dan dapat menangani noise dengan baik. Namun, algoritma ini kurang efisien pada dataset besar karena waktu komputasi yang tinggi, serta sulit diinterpretasi karena hasilnya bergantung pada pemilihan kernel dan parameter yang tepat seperti C, epsilon, dan gamma.

`Model Kalori Terbakar` menggunakan Support Vector Regression dengan parameter standar dengan C sebesar 100 dan epsilon sebesar 0.1 untuk dilakukan regresi. 

<table style="width: 100%; text-align: center;">
  <tbody>
    <tr>
      <td style="background-color: #add8e6; color: #000;">SVR</td>
    </tr>
    <tr>
      <td style="background-color: #e6f7ff; color: #000;"><code>SVR(C=100)</code></td>
    </tr>
  </tbody>
</table>


Model Support Vector Regression pada `Model Kadar Lemak Tubuh` menggunakan teknik Gridsearch untuk menentukan parameternya. Gridsearcgh digunakan karena model awal SVR memiliki nilai metrik akurasi yang sangat rendah. Hasil GridSearch adalah sebagai berikut.

    Best Parameters: {'C': 1, 'epsilon': 0.5, 'gamma': 'scale'}
    Best Score: -0.23618483737773482
    
Parameter model terbaik melalui Gridsearch adalah dengan C sama dengan 1, epsilon sebesar 0.5, dan gamma bernilai 'scale'.

<table>
  <tbody>
    <tr>
      <td style="background-color: #add8e6; color: #000;">SVR</td>
    </tr>
    <tr>
      <td style="background-color: #e6f7ff; color: #000;"><code>SVR(C=1, epsilon=0.5)</code></td>
    </tr>
  </tbody>
</table>

### XGBoost
XGBoost adalah algoritma boosting berbasis pohon keputusan yang membangun model secara iteratif, di mana setiap pohon keputusan  baru dirancang untuk memperbaiki kesalahan dari pohon sebelumnya. XGBoost dilengkapi dengan regularisasi L1 dan L2 yang membantu mengurangi risiko overfitting dan sering kali menghasilkan akurasi tinggi. Algoritma ini juga sangat cepat dan efisien, karena menggunakan teknik optimasi yang canggih. Namun, konfigurasi hyperparameter yang kompleks dan struktur model yang sulit diinterpretasi menjadi kelemahan utama. Selain itu, jika pengaturan model tidak tepat, XGBoost berpotensi mengalami overfitting. 

`Model Kalori Terbakar` dengan metode Extreme Gradient atau XGBoost menggunakan parameter sebagai berikut yaitu n_estimators sebesar 100, learning rate 0.1, max_depth sebesar 6, colsample_bytree sebesar 0.8, dan subsample 0.8 poin.

<table style="width: 100%; text-align: center;">
  <tbody>
    <tr>
      <td style="background-color: #add8e6; color: #000;">XGBRegressor</td>
    </tr>
    <tr>
      <td style="background-color: #e6f7ff; color: #000;"><code>XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=0.8, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.1, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=6, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=100, n_jobs=None,
             num_parallel_tree=None, random_state=None, ...)</code></td>
    </tr>
  </tbody>
</table>

Parameter `Model Kadar Lemak Tubuh` dengan algoritma boosting metode Extreme Gradient atau XGBoost dicari melalui GridSearch karena hasil iterasi pertama model menghasilkan akurasi yang kurang memuaskan.


    Fitting 3 folds for each of 1944 candidates, totalling 5832 fits
    Best Parameters: {'colsample_bytree': 0.8, 'gamma': 0.2, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'reg_alpha': 0, 'reg_lambda': 10.0, 'subsample': 1.0}
    Best Training RMSE: 0.4484218489853021
    Test RMSE: 0.45672201756778374


Parameter yang didapat adalah n_estimators sebesar 50, learning rate 0.1, max_depth sebesar 3, colsample_bytree sebesar 0.8, subsample 1.0 lalu reg_alpha 0, dan reg_lambda sebesar 10.0 poin.

<table>
  <tbody>
    <tr>
      <td style="background-color: #add8e6; color: #000;">XGBRegressor</td>
    </tr>
    <tr>
      <td style="background-color: #e6f7ff; color: #000;"><code>XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=0.8, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=0.2, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.1, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=3, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=50, n_jobs=None,
             num_parallel_tree=None, random_state=None, ...)</code></td>
    </tr>
  </tbody>
</table>





## **Evaluasi**

### Metrik

Setelah itu dipilih dua metrik utama untuk digunakan sebagai pembanding yaitu Adjusted R² dan Mean Squared Error. Keduanya dipilih karena variabel bebas digunakan tidak hanya satu jadi dibutuhkan metrik yang andal dengan jumlah variabel bebas yang lebih dari satu.

**Adjusted R² (Adjusted R-Squared)**

R² mengukur seberapa baik model regresi linier dapat menjelaskan variasi data. Nilainya antara 0 hingga 1, semakin tinggi nilai R², semakin baik model dalam menjelaskan data. Namun, R² bisa meningkat hanya dengan menambahkan variabel bebas ke dalam model, meskipun variabel tersebut mungkin tidak relevan. Karena itu, Adjusted R² digunakan untuk memberikan penilaian yang lebih akurat.

Formula Adjusted R²:

`Adjusted R² = 1 - ((1 - R²) * (n - 1)) / (n - p - 1)`

R²: Koefisien determinasi

n: Jumlah data (observasi)

p: Jumlah variabel bebas

Cara Kerja:

Adjusted R² memperhitungkan jumlah variabel dan jumlah data, membuatnya lebih andal daripada R² saat membandingkan model dengan jumlah variabel yang berbeda.
Jika variabel baru ditambahkan dan relevan, Adjusted R² akan meningkat. Sebaliknya, jika variabel tersebut tidak relevan, Adjusted R² akan menurun, menunjukkan model menjadi lebih kompleks tanpa nilai tambah.
Adjusted R² lebih cocok digunakan untuk membandingkan model yang memiliki jumlah variabel yang berbeda.
Contoh:
Jika Adjusted R² sebuah model adalah 0.85, itu berarti model tersebut dapat menjelaskan 85% variasi data dengan memperhitungkan jumlah variabel dalam model.

**MSE (Mean Squared Error)**

MSE adalah ukuran yang digunakan untuk menghitung rata-rata kesalahan kuadrat antara nilai yang diprediksi oleh model dan nilai aktual. Ini adalah alat yang digunakan untuk mengevaluasi akurasi prediksi model.

Formula MSE:

`MSE = (1 / n) * Σ(yᵢ - ŷᵢ)²`

yᵢ: Nilai aktual (observasi sebenarnya)

ŷᵢ: Nilai prediksi model

n: Jumlah data (observasi)

Cara Kerja:

MSE mengukur rata-rata kuadrat selisih antara nilai yang diprediksi dan nilai yang sebenarnya.
Nilai MSE yang lebih kecil menunjukkan model yang lebih baik, karena perbedaan antara nilai yang diprediksi dan nilai aktual lebih kecil.
MSE sangat sensitif terhadap outlier, karena kesalahan dihitung dalam bentuk kuadrat. Jadi, jika ada data yang sangat berbeda, MSE bisa menjadi lebih besar.
Contoh:
Jika MSE model adalah 4, itu berarti rata-rata kuadrat perbedaan antara nilai prediksi dan nilai aktual adalah 4, menunjukkan model memiliki kesalahan yang relatif besar dibandingkan dengan model dengan MSE lebih kecil.

Selain metrik utama tersebut, Berikut metrik-metrik lain yang digunakan sebagai pembanding.

**R² Score (Coefficient of Determination)**
   
`R² = 1 - ( Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)² )`

Mengukur seberapa baik model menjelaskan variabilitas data target; nilai mendekati 1 menunjukkan model yang baik.

**Root Mean Squared Error (RMSE)**
   
`RMSE = √( Σ(y_i - ŷ_i)² / n )`

Mengukur rata-rata kesalahan prediksi model dalam satuan yang sama dengan data aslinya; semakin kecil, semakin baik.

**Mean Absolute Error (MAE)**
   
`MAE = Σ|y_i - ŷ_i| / n`

Mengukur rata-rata kesalahan absolut antara prediksi dan nilai aktual; mencerminkan kesalahan rata-rata model.

**Explained Variance**
   
`Explained Variance = 1 - ( Var(y - ŷ) / Var(y) )`

Mengukur proporsi varians data target yang dapat dijelaskan oleh model; semakin tinggi, semakin baik.

- y_i: nilai sebenarnya
- ŷ_i: nilai prediksi
- ȳ: mean dari nilai sebenarnya
- n: jumlah data
- Var(x): varians dari x


### Grafik

Grafik yang disajikan dalam evaluasi ini adalah grafik antara hasil prediksi dengan nilai asli yang terdapat dalam data. seperti contoh berikut

![png](gambar_files/gambar_97_2.png)

gambar tersebut menunjukan titik-titik hasil plot antara nilai prediksi pada sumbu y dan nilai ebenarnya pada sumbu X. Ketika nilai keduanya sama maka akan memenuhi garis diagonal 45 derajat. Jika nilai prediksi lebih kecil dari nilai sebenarnya maka titik akan berada diatas garis. Dan apabila nilai prediksi lebih besar dari nilai sebenarnya, titik tersebut akan berada di bawah garis. Grafik tersebut berguna untuk melihat persebaran data yang melenceng dari nilai prediksi


### Model Kalori Terbakar

#### Random Forest

    Model Performance (RF):
                               0
    Model                     RF
    R2 Score            0.970312
    Adjusted R2         0.967643
    RMSE                0.182629
    MAE                  0.13895
    MSE                 0.033353
    Explained Variance  0.970519

![png](gambar_files/gambar_97_2.png)
    
Model yang dihasilkan cukup bagus dengan Adjusted Rsquared sebesar 0.9676 dan MSE 0.0334 yang berada dibawah 0.1 nilainya.

#### KNN

    Model Performance (KNN):
                               0
    Model                    KNN
    R2 Score            0.849978
    Adjusted R2         0.836493
    RMSE                0.410542
    MAE                 0.332696
    MSE                 0.168544
    Explained Variance  0.850501
    
![png](gambar_files/gambar_102_1.png)
    

Model ini tidak sebagus model sebelumnya namun masih cukup bagus di Adjusted Rsquared sebesar 0.8365 dan nilai MSE di atas 0.1 yaitu 0.1685 pada model ini.

#### SVR

    
    Model Performance (SVR):
                               0
    Model                    SVR
    R2 Score            0.972084
    Adjusted R2         0.969575
    RMSE                0.177094
    MAE                 0.127105
    MSE                 0.031362
    Explained Variance  0.972092



    
![png](gambar_files/gambar_107_1.png)
    


Hasil dari model ini sangat baik dengan Adjusted Rsquared sebesar 0.9696 dan MSE 0.0314.

#### XGBoost
    
    Model Performance (XGBoost):
                               0
    Model                XGBoost
    R2 Score            0.986622
    Adjusted R2         0.985419
    RMSE                0.122597
    MAE                  0.08944
    MSE                  0.01503
    Explained Variance  0.986663



    
![png](gambar_files/gambar_112_1.png)
    


Hasilnya sangat bagus dengn Adjusted Rsquares sebesar 0.9854 dan MSE yang sangat kecil yaitu 0.0150 jauh dibawah model lainnya.

#### Model Terbaik




         Model  R2 Score  Adjusted R2      RMSE       MAE       MSE  \
    0       RF  0.970312     0.967643  0.182629  0.138950  0.033353   
    1      KNN  0.849978     0.836493  0.410542  0.332696  0.168544   
    2      SVR  0.972084     0.969575  0.177094  0.127105  0.031362   
    3  XGBoost  0.986622     0.985419  0.122597  0.089440  0.015030   
    
       Explained Variance  
    0            0.970519  
    1            0.850501  
    2            0.972092  
    3            0.986663  



    
![png](gambar_files/gambar_115_1.png)
    


Model terbaik untuk variabel Kalori Terbakar adalah dengan Metode XGBoost yaitu dengan Adjusted Rsquared 0.9854.


### Model Kadar Lemak Tubuh

#### Random Forest


    Model Performance (RF):
                               0
    Model                     RF
    R2 Score            0.804363
    Adjusted R2         0.789147
    RMSE                0.445087
    MAE                 0.377017
    MSE                 0.198102
    Explained Variance   0.80439

    
![png](gambar_files/gambar_124_2.png)
    
Model yang dihasilkan kurang bagus dengan Adjusted Rsquared sebesar 0.7891 dan MSE 0.1981 yang relatif besar galatnya.

#### KNN



    Model Performance (KNN):
                               0
    Model                    KNN
    R2 Score            0.776418
    Adjusted R2         0.759029
    RMSE                0.475813
    MAE                 0.405382
    MSE                 0.226398
    Explained Variance  0.778555

![png](gambar_files/gambar_129_1.png)
    

Model ini lebih buruk dari model sebelumnya namun masih cukup di Adjusted Rsquared sebesar 0.7590 dan nilai MSE tinggi di atas 0.1 yaitu 0.2264 pada model ini.

#### SVR

    
    Model Performance (SVR):
                               0
    Model                    SVR
    R2 Score            0.765147
    Adjusted R2         0.746881
    RMSE                0.487659
    MAE                 0.417695
    MSE                 0.237811
    Explained Variance  0.767112

    
![png](gambar_files/gambar_136_1.png)
    


Hasil dari model ini kurang baik dengan Adjusted Rsquared sebesar 0.7469 dan MSE 0.2378.

#### XGBoost

    
    Model Performance (XGBoost):
                               0
    Model                XGBoost
    R2 Score               0.794
    Adjusted R2         0.777978
    RMSE                0.456722
    MAE                 0.393461
    MSE                 0.208595
    Explained Variance  0.794144



    
![png](gambar_files/gambar_143_1.png)
    


Hasilnya agak sedikit lebih baik dengn Adjusted Rsquares sebesar 0.777978 dan MSE yang masih diatas 0.1 yaitu 0.2086 poin.

#### Model Terbaik



         Model  R2 Score  Adjusted R2      RMSE       MAE       MSE  \
    0       RF  0.804363     0.789147  0.445087  0.377017  0.198102   
    1      KNN  0.776418     0.759029  0.475813  0.405382  0.226398   
    2      SVR  0.765147     0.746881  0.487659  0.417695  0.237811   
    3  XGBoost  0.794000     0.777978  0.456722  0.393461  0.208595   
    
       Explained Variance  
    0            0.804390  
    1            0.778555  
    2            0.767112  
    3            0.794144  



    
![png](gambar_files/gambar_146_1.png)
    

Model terbaik untuk variabel Kadar Lemak Tubuh ternyata adalah model Random Forest dengan Adjusted Rsquared sebesar 0.7891 terbesar dari yang lain.

## Kesimpulan

Kesimpulan berdasarkan hasil dari *goal* yang telah dicapai:  

1. **Model Prediksi Kalori Terbakar**  
   Model prediksi kalori terbakar menunjukkan kinerja yang sangat baik dengan *adjusted R²* sebesar **0.9854** dengan menggunakan algoritma XGBoost, menandakan bahwa model mampu menjelaskan sebagian besar variabilitas dalam jumlah kalori yang terbakar. Hal ini menunjukkan potensi yang tinggi untuk mengimplementasikan model ini dalam memberikan rekomendasi latihan yang efisien.


2. **Model Prediksi Kadar Lemak Tubuh**  
   Model prediksi kadar lemak tubuh memiliki *adjusted R²* sebesar **0.7891** dengan mengunakan algoritma Random Forest, menunjukkan bahwa model ini cukup baik dalam memprediksi kadar lemak tubuh. Namun, masih ada ruang untuk meningkatkan akurasi model, mungkin dengan menambahkan variabel atau data yang lebih relevan.  

3. **Preferensi Latihan Berdasarkan Gender**  
   Analisis tidak menemukan perbedaan yang signifikan dalam preferensi latihan antara laki-laki dan perempuan. Hal ini mengindikasikan bahwa program latihan dapat dirancang dengan pendekatan yang lebih universal tanpa perlu segregasi berdasarkan gender.  

4. **Preferensi Berdasarkan Tingkat Kemahiran**  
   Tidak ada hubungan signifikan yang ditemukan antara tingkat kemahiran dan preferensi latihan. Artinya, preferensi latihan kemungkinan lebih dipengaruhi oleh faktor lain, seperti tujuan kebugaran individu atau akses ke fasilitas, daripada tingkat pengalaman.  


Secara keseluruhan, model prediksi yang kuat untuk kalori terbakar dan kadar lemak tubuh telah berhasil dibuat. Hasil analisis preferensi menunjukkan bahwa pendekatan universal dapat diambil tanpa perlu memperhatikan perbedaan gender atau tingkat kemahiran. Langkah selanjutnya dapat difokuskan pada pengembangan program latihan berbasis data, peningkatan model kadar lemak tubuh, atau eksplorasi faktor-faktor lain yang mungkin memengaruhi preferensi individu.



### Referensi

- Gough, A., et al. (2018). Personalized Fitness: Trends in the Digital Fitness Industry. *Journal of Health & Wellness*.

- McAuley, E., et al. (2011). Social Support and Self-Efficacy in Exercise. *Health Psychology*.

- World Health Organization. (2020). *Physical Activity*. Retrieved from [WHO](https://www.who.int/news-room/fact-sheets/detail/physical-activity).

- Tan, J. S. A., Che Embi, Z., & Hashim, N. (2024). Comparison of Machine Learning Methods for Calories Burn Prediction. *Journal of Informatics and Web Engineering*, 3(1), 182-191. [https://doi.org/10.33093/jiwe.2024.3.1.12](https://doi.org/10.33093/jiwe.2024.3.1.12)

- Kadam, A., Shrivastava, A., Pawar, S. K., Patil, V. H., Michaelson, J., & Singh, A. (n.d.). *Calories Burned Prediction Using Machine Learning*. IEEE. Retrieved from [Calories Burn Prediction](https://hossainlab.github.io/projects/Calories_Burnt/02_Calories%20Burnt%20Prediction.html)
