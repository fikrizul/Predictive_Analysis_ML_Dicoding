# Laporan Proyek Machine Learning - Fikri Zulfialdi

## Domain Proyek

Optimalisasi aktivitas latihan di pusat kebugaran telah menjadi fokus utama dalam upaya meningkatkan kesehatan masyarakat modern, terutama dalam konteks efektivitas pembakaran kalori. Pemahaman mendalam tentang aktivitas yang menghasilkan pembakaran kalori optimal tidak hanya penting bagi individu yang mengejar tujuan kebugaran pribadi, tetapi juga memiliki implikasi signifikan terhadap kesehatan masyarakat secara keseluruhan. Data dari Organisasi Kesehatan Dunia (WHO, 2020) menunjukkan bahwa tingkat aktivitas fisik yang memadai dapat secara substansial mengurangi risiko berbagai penyakit kronis, namun mayoritas populasi global masih belum mencapai tingkat aktivitas fisik yang direkomendasikan. Tantangan ini semakin diperumit oleh keterbatasan waktu yang dihadapi masyarakat modern, mendorong kebutuhan akan pemahaman yang lebih baik tentang efektivitas berbagai jenis latihan dalam memaksimalkan pembakaran kalori dalam waktu yang tersedia.

Penelitian yang dilakukan oleh Gough et al. (2018) mengungkapkan adanya pergeseran signifikan dalam preferensi masyarakat terhadap program latihan yang lebih efisien namun tetap efektif, mencerminkan kebutuhan akan optimalisasi waktu dalam konteks kesehatan modern. Tren ini memperkuat pentingnya mengidentifikasi dan memahami aktivitas gym yang memberikan manfaat maksimal dalam durasi minimal, memungkinkan individu untuk mencapai tujuan kesehatan mereka meskipun menghadapi kendala waktu. Lebih lanjut, McAuley et al. (2011) menekankan peran krusial faktor psikososial dalam efektivitas latihan, menunjukkan bahwa dukungan sosial dan kepercayaan diri secara signifikan mempengaruhi intensitas dan konsistensi latihan. Temuan ini menggarisbawahi pentingnya mempertimbangkan tidak hanya aspek fisiologis dari pembakaran kalori, tetapi juga konteks psikologis dan sosial yang mempengaruhi efektivitas latihan secara keseluruhan.

Dalam konteks kesehatan masyarakat yang lebih luas, pemahaman tentang efektivitas pembakaran kalori dalam berbagai aktivitas gym memiliki implikasi penting untuk pengembangan strategi intervensi kesehatan yang lebih efektif. Hal ini menjadi semakin relevan mengingat meningkatnya prevalensi penyakit terkait gaya hidup sedenter, seperti obesitas, diabetes, dan penyakit kardiovaskular. Optimalisasi program latihan berdasarkan pemahaman yang lebih baik tentang efektivitas pembakaran kalori dapat membantu mengatasi tantangan kesehatan ini dengan lebih efektif, sambil mempertimbangkan keterbatasan waktu dan sumber daya yang dihadapi masyarakat modern. Dengan demikian, penelitian tentang efektivitas pembakaran kalori dalam aktivitas gym tidak hanya berkontribusi pada pengembangan program kebugaran yang lebih efisien, tetapi juga berperan penting dalam upaya yang lebih luas untuk meningkatkan kesehatan masyarakat dan mengurangi beban penyakit kronis.

Penerapan machine learning memungkinkan identifikasi perilaku gym yang paling efektif, dengan algoritma seperti k-Nearest Neighbors (KNN), Random Forest, dan Boosting. KNN dapat menyarankan latihan berdasarkan karakteristik individu, sementara Random Forest menggabungkan berbagai faktor untuk prediksi pembakaran kalori yang akurat. Boosting meningkatkan ketepatan rekomendasi latihan dengan terus belajar dari kesalahan sebelumnya. Dengan model ini, pusat kebugaran bisa memberikan rekomendasi yang lebih dipersonalisasi, membantu pengguna mencapai tujuan kebugaran secara efisien sekaligus mendukung peningkatan kesehatan masyarakat melalui gaya hidup aktif. Data yang digunakan diambil dari kaggle yang bisa diakses dari link [berikut](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset/data)

Referensi:
- World Health Organization. (2020). Physical Activity. WHO Fact Sheets.
- Gough, A., et al. (2018). Personalized Fitness: Trends in the Digital Fitness Industry. Journal of Health & Wellness.
- McAuley, E., et al. (2011). Social Support and Self-Efficacy in Exercise. Health Psychology.

- Here is the updated reference list, including the new citation:

**References:**

- Gough, A., et al. (2018). Personalized Fitness: Trends in the Digital Fitness Industry. *Journal of Health & Wellness*.
  
- McAuley, E., et al. (2011). Social Support and Self-Efficacy in Exercise. *Health Psychology*.

- World Health Organization. (2020). *Physical Activity*. Retrieved from [WHO](https://www.who.int/news-room/fact-sheets/detail/physical-activity).

- Tan, J. S. A., Che Embi, Z., & Hashim, N. (2024). Comparison of Machine Learning Methods for Calories Burn Prediction. *Journal of Informatics and Web Engineering*, 3(1), 182-191. doi: [10.33093/jiwe.2024.3.1.12](https://doi.org/10.33093/jiwe.2024.3.1.12).

- Kadam, A., Shrivastava, A., Pawar, S. K., Patil, V. H., Michaelson, J., & Singh, A. (n.d.). *Calories Burned Prediction Using Machine Learning*. IEEE. Retrieved from [Calories Burn Prediction](https://hossainlab.github.io/projects/Calories_Burnt/02_Calories%20Burnt%20Prediction.html).


Dataset ini mencakup profil kebugaran individu, meliputi detail demografis (usia, jenis kelamin), komposisi tubuh (berat badan, tinggi badan, BMI, persentase lemak), metrik detak jantung (Max_BPM, Avg_BPM, Resting_BPM), dan kebiasaan olahraga (Jenis Olahraga, Durasi Sesi, Kalori Terbakar, Frekuensi Olahraga). **BMI** dan **Persentase Lemak** memberikan gambaran tentang komposisi tubuh, dengan **Persentase Lemak** yang biasanya memberikan gambaran yang lebih akurat dibandingkan BMI, terutama untuk individu dengan massa otot yang tinggi. Metrik detak jantung menyoroti kebugaran kardiovaskular, di mana **Resting_BPM** sering kali lebih rendah pada individu yang lebih fit.

Data olahraga menunjukkan intensitas dan preferensi, dengan aktivitas berintensitas tinggi (seperti HIIT atau Kardio) yang cenderung membakar lebih banyak kalori dan memiliki **Avg_BPM** lebih tinggi dibandingkan dengan latihan berintensitas rendah seperti Yoga. **Asupan Air** dan **Tingkat Pengalaman** menambah kedalaman informasi, menunjukkan kebiasaan hidrasi dan tingkat keakraban dengan kebugaran, yang dapat memengaruhi hasil latihan dan detak jantung saat istirahat. Dataset ini memungkinkan pemahaman yang luas tentang tingkat kebugaran individu dan memberikan wawasan yang berguna untuk personalisasi rencana kebugaran dan kesehatan.

These references should comprehensively support your essay on using machine learning techniques for predicting calorie burn during gym activities.

## Business Understanding
