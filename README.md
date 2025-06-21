Forward and Backward Propagation ile Yapay Sinir Ağı Eğitimi
Bu proje, YZM212 Makine Öğrenmesi dersi kapsamında, temel bir Yapay Sinir Ağı (ANN) modeli geliştirmek amacıyla hazırlanmıştır. Model, ileri (forward) ve geri (backward) yayılım algoritmaları kullanılarak sıfırdan oluşturulmuş, yalnızca NumPy gibi temel kütüphanelerle gerçekleştirilmiştir. Hazır makine öğrenmesi kütüphaneleri (scikit-learn, Keras, PyTorch vb.) kullanılmamıştır.

 Veri Seti ve Ön İşleme
Kullanılan Veri Seti
Proje kapsamında Iris veri seti kullanılmıştır. Bu veri seti 3 farklı iris çiçeği türüne (Setosa, Versicolor, Virginica) ait 4 özellik içermektedir:

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

Ön İşleme Adımları
Eksik Veri Kontrolü: Veri setinde eksik değer bulunmamaktadır.

Etiket Kodlama: Çiçek türleri LabelBinarizer kullanılarak one-hot encoding ile dönüştürülmüştür.

Özellik Ölçekleme: Tüm özellikler Min-Max Normalizasyonu ile [0, 1] aralığına çekilmiştir.

Veri Ayrımı: Veri seti eğitim (%80) ve test (%20) olarak ayrılmıştır.

 Model Mimarisi
Yapay sinir ağı aşağıdaki gibi üç katmandan oluşmaktadır:

Katman	Nöron Sayısı	Aktivasyon Fonksiyonu
Giriş Katmanı	4 (özellik sayısı)	-
Gizli Katman	8 (örnek)	Sigmoid
Çıkış Katmanı	3 (sınıf sayısı)	Softmax

Not: Gizli katman sayısı ve nöron sayısı parametre olarak değiştirilebilir.

Aktivasyon Fonksiyonları
Sigmoid:
Formül: σ(x) = 1 / (1 + e^(-x))

Softmax:
Formül: softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)

 Modelin Çalışma Prensibi
İleri Yayılım (Forward Propagation)

Girdi, ağırlıklarla çarpılır ve bias eklenir.

Gizli katmandaki çıktılar Sigmoid fonksiyonu ile aktive edilir.

Çıkış katmanına iletilen veriler softmax ile olasılık değerlerine dönüştürülür.

Hata Hesaplama

Kayıp fonksiyonu olarak Cross-Entropy Loss kullanılmıştır:

L = -Σ(y_true × log(y_pred))

Geri Yayılım (Backpropagation)

Hata, zincir kuralı kullanılarak katmanlara geri yayılır.

Ağırlık ve bias'ların gradyanları hesaplanır ve güncellenir:

weight = weight - learning_rate × gradient

Gradyan İniş (Gradient Descent)

Ağırlıklar her epoch sonunda güncellenir.

Öğrenme oranı (learning_rate = 0.1) olarak belirlenmiştir.

 Eğitim Süreci
Model 1000 epoch boyunca eğitilmiştir. Her epoch sonunda eğitim kaybı (loss) hesaplanmış ve loss vs epoch grafiği oluşturulmuştur.

Eğitim Parametreleri
Parametre	Değer
Epoch	1000
Öğrenme Oranı	0.1
Gizli Nöron Sayısı	8
Aktivasyon	Sigmoid / Softmax

 Sonuçlar
Kayıp Değeri (Loss)

Eğitim başlangıcında (Epoch 1): yaklaşık 1.16

Eğitim sonunda (Epoch 1000): yaklaşık 0.04

Kayıp eğrisi sürekli azalmış ve modelin başarıyla öğrenme gerçekleştirdiği görülmüştür.

 Grafikte detaylı olarak gösterilmiştir:
![accuracy_plot](https://github.com/user-attachments/assets/fe03d63a-4f9e-4ad1-b544-66aa4678af38)



Karmaşıklık Matrisi (Confusion Matrix)

Gerçek \ Tahmin	Setosa	Versicolor	Virginica
Setosa	10	0	0
Versicolor	0	10	0
Virginica	0	0	10

Toplam doğru sınıflandırma: 30 / 30

 Görsel hali:
![confusion_matrix](https://github.com/user-attachments/assets/2774718e-f6bd-46f3-94eb-42db254660de)

Test Doğruluğu (Accuracy)

Doğru sınıflandırma oranı: %100.00

(30 test örneğinin tamamı doğru sınıflandırılmıştır.)


 Proje Dosya Yapısı

5.ForwardAndBackwardPropagation/
├── forward_backward_nn.ipynb     # Ana model dosyası (Jupyter Notebook)
├── dataset.csv                   # Kullanılan veri seti
├── accuracy_plot.png             # Loss vs Epoch grafiği
├── confusion_matrix.png          # Karmaşıklık matrisi görseli
├── Readme.md                     # Bu açıklama dosyası
├── requirements.txt              # Gerekli kütüphaneler listesi
 Kullanılan Kütüphaneler
python
Kopyala
Düzenle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
Not: scikit-learn yalnızca veri bölme, etiketleme ve değerlendirme metrikleri için kullanılmıştır. Model sıfırdan elle yazılmıştır.


 Kaynakça
Brownlee, J. (2018). Neural Networks from Scratch in Python. Machine Learning Mastery.
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

https://www.youtube.com/watch?v=f26KI43FK58

https://github.com/git-guides/#learning-git-basics
