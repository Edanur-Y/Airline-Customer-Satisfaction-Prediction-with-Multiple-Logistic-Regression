Çoklu Lojistik Regresyon ile Havayolu Müşteri Memnuniyeti Analizi
================
Edanur Yılmaz,
2024-01-18

## İçerik
- [Verinin İncelenmesi ve
  Hazırlanması](#verinin-i̇ncelenmesi-ve-hazırlanması)
- [Modeli Oluşturma ve Yorumlama](#modeli-oluşturma-ve-yorumlama)
- [Model Üzerinden Tahmin](#model-üzerinden-tahmin)
- [Optimal Eşik Değeri](#optimal-eşik-değeri)
- [Precision ve Recall](#precision-ve-recall)
- [F1 Score](#f1-score)
- [ROC Curve ve AUC (Area Under
  Curve)](#roc-curve-ve-auc-area-under-curve)
- [caret Paketi ile Performans
  Yorumlama](#caret-paketi-ile-performans-yorumlama)

Gerekli paketler:

``` r
library(ggpubr)
library(mice)
library(broom)
library(dplyr)
library(magrittr)
library(caret)
library(InformationValue)
library(modelr)
library(ROCR)
library(pROC)
```

## Verinin İncelenmesi ve Hazırlanması
<sup>[İçeriğe dön.](#İçerik)</sup>

**Veriyi Tanıma**

Kaynak veri seti: [Airlines Customer
satisfaction](https://www.kaggle.com/datasets/sjleshrac/airlines-customer-satisfaction/data)

``` r
#View(Airlines_Data)
names(Airlines_Data)
```

    ##  [1] "satisfaction"                      "Gender"                           
    ##  [3] "Customer.Type"                     "Age"                              
    ##  [5] "Type.of.Travel"                    "Class"                            
    ##  [7] "Flight.Distance"                   "Seat.comfort"                     
    ##  [9] "Departure.Arrival.time.convenient" "Food.and.drink"                   
    ## [11] "Gate.location"                     "Inflight.wifi.service"            
    ## [13] "Inflight.entertainment"            "Online.support"                   
    ## [15] "Ease.of.Online.booking"            "On.board.service"                 
    ## [17] "Leg.room.service"                  "Baggage.handling"                 
    ## [19] "Checkin.service"                   "Cleanliness"                      
    ## [21] "Online.boarding"                   "Departure.Delay.in.Minutes"       
    ## [23] "Arrival.Delay.in.Minutes"

Müşteri memnuniyeti ölçmek için hazırlanmış bir veri setini inceliyoruz.
Veri setinde Cinsiyet, Müşteri Tipi(2), Yaş(5-85), Seyehat Tipi(2),
Sınıf Türü(Class)(3), Uçuş Mesafesi(0-7000), Kalkış-İniş Gecikme
Süreleri(0-1600dk) ve içerisinde Check-in Servisi, Koltuk Konforu gibi
değişkenler olan 0 ila 5 arasında değerlendirilmiş 14 farklı değişkenle
birlikte toplam 23 değişken bulunmakta.

Bu bağımsız değişkenlere dayalı olarak müşterilerin memnuniyetini
(***satisfaction***) tahmin eden bir çoklu lojistik regresyon modeli
oluşturacağız.

**Kayıp Gözlemler**

``` r
sum(is.na(Airlines_Data))
```

    ## [1] 393

``` r
names(which(colSums(is.na(Airlines_Data)) > 0))
```

    ## [1] "Arrival.Delay.in.Minutes"

Veri seti incelendiğinde ***Arrival.Delay.in.Minutes*** değişkeninde
kayıp gözlemler olduğu görülüyor.

``` r
sum(na.omit(Airlines_Data$Arrival.Delay.in.Minutes==0))
```

    ## [1] 72753

``` r
nrow(Airlines_Data)
```

    ## [1] 129880

Bu değişkenin gözlemlerinin yarısından çoğu 0 değeri almakta, bu durum
kayıp verileri doldurmak için kullanacağımız değerleri etkileyecektir.
Gözlem sayımız da yeterince büyük olduğundan kayıp gözlemleri
çıkarıyoruz.

``` r
Airlines_Data<-na.omit(Airlines_Data)
sum(is.na(Airlines_Data))
```

    ## [1] 0

***satisfaction*** değişkeninin her iki alt kategorisinde birden yer
almayan bir bağımsız değişken var mı diye bakıyoruz. Genel olarak bir
sıkıntı gözükmüyor. 0-5 arası değer alan değişkenlerden bazılarının 0
alt kategorilerinde farklar açılsa da üstünde durulması gerektiğini
düşünmüyorum.

``` r
variables<-subset(Airlines_Data, select = c(Gender, Customer.Type, Type.of.Travel, Class, Seat.comfort, Ease.of.Online.booking))

sat_status<-function(variables){
  for (variable in variables){
    print(xtabs(~ satisfaction + variable, data=Airlines_Data))
  }
}

sat_status(variables)
```

    ##               variable
    ## satisfaction   Female  Male
    ##   dissatisfied  22904 35701
    ##   satisfied     42799 28083
    ##               variable
    ## satisfaction   disloyal Customer Loyal Customer
    ##   dissatisfied             18026          40579
    ##   satisfied                 5688          65194
    ##               variable
    ## satisfaction   Business travel Personal Travel
    ##   dissatisfied           37238           21367
    ##   satisfied              52207           18675
    ##               variable
    ## satisfaction   Business   Eco Eco Plus
    ##   dissatisfied    18013 35219     5373
    ##   satisfied       43977 22898     4007
    ##               variable
    ## satisfaction       0     1     2     3     4     5
    ##   dissatisfied    10 11466 18396 18734  9858   141
    ##   satisfied     4771  9416 10249 10362 18457 17627
    ##               variable
    ## satisfaction       0     1     2     3     4     5
    ##   dissatisfied    18 10815 14192 14366 11233  7981
    ##   satisfied        0  2582  5695  7978 28574 26053

## Modeli Oluşturma ve Yorumlama
<sup>[İçeriğe dön.](#İçerik)</sup>

***satisfaction*** değişkeninin aldığı “satisfied”(memnun) ve
“dissatisfied”(memnun değil) değerlerinin miktarlarını inceliyoruz.

``` r
dataSatisfied<-Airlines_Data %>% filter(satisfaction == "satisfied")
dataDissatisfied<-Airlines_Data %>% filter(satisfaction == "dissatisfied")
```

![](Çoklu-Lojistik-Regresyon-ile-Havayolu-Müşteri-Memnuniyeti-Analizi_files/figure-gfm/datasatisfactionplot-1.png)<!-- -->
***satisfaction*** değişkeninin “satisfied” değerini “dissatisfied”
değerinden daha fazla aldığını görebiliyoruz. Her bir alt kategornin
yaklaşık oranlarda sisteme dahil olabilmesi adına bu değerlerin
miktarlarını düzenleyeceğiz.

Eğitim setine dahil olacak değişkenlerin miktarını daha az sayıda olan
alt kategoriye göre ayarlıyoruz.

``` r
set.seed(44)
dataSatisfiedIndex<-sample(1:nrow(dataSatisfied),size=0.75*nrow(dataDissatisfied))
set.seed(44)
dataDissatisfieddIndex<-sample(1:nrow(dataDissatisfied),size=0.75*nrow(dataDissatisfied))
```

``` r
trainSatisfied<-dataSatisfied[dataSatisfiedIndex,]
trainDissatisfied<-dataDissatisfied[dataDissatisfieddIndex,]

trainset<-rbind(trainSatisfied, trainDissatisfied)
```

![](Çoklu-Lojistik-Regresyon-ile-Havayolu-Müşteri-Memnuniyeti-Analizi_files/figure-gfm/trainsatisfactionplot-1.png)<!-- -->
Geriye kalan verilerle de test setini oluşturuyoruz.

``` r
testSatisfied<-dataSatisfied[-dataSatisfiedIndex,]
testDissatisfied<-dataDissatisfied[-dataDissatisfieddIndex,]
```

``` r
testset<-rbind(testSatisfied, testDissatisfied)
table(testset$satisfaction)
```

    ## 
    ## dissatisfied    satisfied 
    ##        14652        26929

Kategorik olan ***satisfaction*** değişkenini faktör olarak
biçimlendiriyoruz.

``` r
trainset$satisfaction <- factor(trainset$satisfaction)
class(trainset$satisfaction)
```

    ## [1] "factor"

Modeli oluşturuyoruz.

``` r
model1<- glm(satisfaction ~ ., family = "binomial", data=trainset)
summary(model1)
```

    ## 
    ## Call:
    ## glm(formula = satisfaction ~ ., family = "binomial", data = trainset)
    ## 
    ## Coefficients:
    ##                                     Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                       -7.1585302  0.0802495 -89.203  < 2e-16 ***
    ## GenderMale                        -0.9375447  0.0199416 -47.015  < 2e-16 ***
    ## Customer.TypeLoyal Customer        1.9808660  0.0305337  64.875  < 2e-16 ***
    ## Age                               -0.0069261  0.0006934  -9.989  < 2e-16 ***
    ## Type.of.TravelPersonal Travel     -0.7790824  0.0283537 -27.477  < 2e-16 ***
    ## ClassEco                          -0.7191619  0.0256867 -27.997  < 2e-16 ***
    ## ClassEco Plus                     -0.7984791  0.0397274 -20.099  < 2e-16 ***
    ## Flight.Distance                   -0.0001020  0.0000104  -9.804  < 2e-16 ***
    ## Seat.comfort                       0.2722834  0.0111593  24.400  < 2e-16 ***
    ## Departure.Arrival.time.convenient -0.2019634  0.0083153 -24.288  < 2e-16 ***
    ## Food.and.drink                    -0.1968002  0.0114039 -17.257  < 2e-16 ***
    ## Gate.location                      0.1125481  0.0093584  12.026  < 2e-16 ***
    ## Inflight.wifi.service             -0.0708620  0.0107088  -6.617 3.66e-11 ***
    ## Inflight.entertainment             0.7038131  0.0102298  68.800  < 2e-16 ***
    ## Online.support                     0.0994288  0.0109908   9.047  < 2e-16 ***
    ## Ease.of.Online.booking             0.2336402  0.0142137  16.438  < 2e-16 ***
    ## On.board.service                   0.3156673  0.0100856  31.299  < 2e-16 ***
    ## Leg.room.service                   0.2278145  0.0085695  26.584  < 2e-16 ***
    ## Baggage.handling                   0.1025284  0.0113675   9.019  < 2e-16 ***
    ## Checkin.service                    0.2939886  0.0084379  34.841  < 2e-16 ***
    ## Cleanliness                        0.0838232  0.0118676   7.063 1.63e-12 ***
    ## Online.boarding                    0.1572846  0.0120497  13.053  < 2e-16 ***
    ## Departure.Delay.in.Minutes         0.0027209  0.0009743   2.793  0.00523 ** 
    ## Arrival.Delay.in.Minutes          -0.0081974  0.0009598  -8.541  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 121864  on 87905  degrees of freedom
    ## Residual deviance:  67885  on 87882  degrees of freedom
    ## AIC: 67933
    ## 
    ## Number of Fisher Scoring iterations: 5

Model sonuçları incelendiğinde ***Departure.Delay.in.Minutes***
değişkeninin anlamlılığının diğer değişkenlerden düşük olduğunu
görüyoruz. Null deviance ve Residual deviance farkına baktığımızda da
eklenen değişkenlerin modeli iyileştirdiğini görüyoruz.

Lojistik regresyon varsayımı olan null(boş) hipotezine göre tüm
katsayılar sıfıra eşittir. Eklediğimiz değişkenlerin önemi olduğunu
kanıtlamak için p değerini hesapladığımızda 0.5’ten küçük olduğunu
görüyoruz. Dolayısıyla null hipotezini reddedebiliriz.

``` r
1-pchisq(121864 - 67885, 87905-87882)
```

    ## [1] 0

``` r
anova(model1)
```

    ## Analysis of Deviance Table
    ## 
    ## Model: binomial, link: logit
    ## 
    ## Response: satisfaction
    ## 
    ## Terms added sequentially (first to last)
    ## 
    ## 
    ##                                   Df Deviance Resid. Df Resid. Dev
    ## NULL                                              87905     121864
    ## Gender                             1   3993.1     87904     117870
    ## Customer.Type                      1   8602.3     87903     109268
    ## Age                                1    157.5     87902     109111
    ## Type.of.Travel                     1   3924.3     87901     105186
    ## Class                              2   4496.6     87899     100690
    ## Flight.Distance                    1    330.5     87898     100359
    ## Seat.comfort                       1   5629.3     87897      94730
    ## Departure.Arrival.time.convenient  1   2247.8     87896      92482
    ## Food.and.drink                     1    422.1     87895      92060
    ## Gate.location                      1    129.2     87894      91931
    ## Inflight.wifi.service              1   2335.5     87893      89595
    ## Inflight.entertainment             1   9431.0     87892      80164
    ## Online.support                     1   1208.4     87891      78956
    ## Ease.of.Online.booking             1   4948.9     87890      74007
    ## On.board.service                   1   2789.4     87889      71218
    ## Leg.room.service                   1    980.0     87888      70238
    ## Baggage.handling                   1    263.0     87887      69975
    ## Checkin.service                    1   1417.9     87886      68557
    ## Cleanliness                        1     54.2     87885      68503
    ## Online.boarding                    1    150.6     87884      68352
    ## Departure.Delay.in.Minutes         1    392.8     87883      67959
    ## Arrival.Delay.in.Minutes           1     73.7     87882      67885

Her değişkenin, eklendiğinde Residual deviance değerini ne kadar
değiştirdiğini inceleyebileceğimiz bir tablo elde ediyoruz.

``` r
varImp(model1)
```

    ##                                     Overall
    ## GenderMale                        47.014594
    ## Customer.TypeLoyal Customer       64.874770
    ## Age                                9.989142
    ## Type.of.TravelPersonal Travel     27.477232
    ## ClassEco                          27.997475
    ## ClassEco Plus                     20.098977
    ## Flight.Distance                    9.804356
    ## Seat.comfort                      24.399618
    ## Departure.Arrival.time.convenient 24.288277
    ## Food.and.drink                    17.257340
    ## Gate.location                     12.026459
    ## Inflight.wifi.service              6.617201
    ## Inflight.entertainment            68.800082
    ## Online.support                     9.046576
    ## Ease.of.Online.booking            16.437699
    ## On.board.service                  31.298648
    ## Leg.room.service                  26.584332
    ## Baggage.handling                   9.019441
    ## Checkin.service                   34.841316
    ## Cleanliness                        7.063194
    ## Online.boarding                   13.053024
    ## Departure.Delay.in.Minutes         2.792732
    ## Arrival.Delay.in.Minutes           8.540578

Değişkenlerin önem seviyelerini gösteren bu dataframe’i incelediğimzde
***Inflight.entertainment*** değişkeninin en fazla etkiye sahip olduğunu
ve ***Departure.Delay.in.Minutes*** değişkeninin en az etkiye sahip
olduğunu görebiliyoruz.

``` r
exp(coef(model1))
```

    ##                       (Intercept)                        GenderMale 
    ##                      0.0007781975                      0.3915881074 
    ##       Customer.TypeLoyal Customer                               Age 
    ##                      7.2490178093                      0.9930978605 
    ##     Type.of.TravelPersonal Travel                          ClassEco 
    ##                      0.4588268313                      0.4871603776 
    ##                     ClassEco Plus                   Flight.Distance 
    ##                      0.4500128733                      0.9998979995 
    ##                      Seat.comfort Departure.Arrival.time.convenient 
    ##                      1.3129591006                      0.8171248414 
    ##                    Food.and.drink                     Gate.location 
    ##                      0.8213547075                      1.1191261350 
    ##             Inflight.wifi.service            Inflight.entertainment 
    ##                      0.9315904121                      2.0214459725 
    ##                    Online.support            Ease.of.Online.booking 
    ##                      1.1045397762                      1.2631898764 
    ##                  On.board.service                  Leg.room.service 
    ##                      1.3711739491                      1.2558523558 
    ##                  Baggage.handling                   Checkin.service 
    ##                      1.1079687770                      1.3417686614 
    ##                       Cleanliness                   Online.boarding 
    ##                      1.0874366306                      1.1703286613 
    ##        Departure.Delay.in.Minutes          Arrival.Delay.in.Minutes 
    ##                      1.0027245939                      0.9918361345

Bağımsız değişkenlerin katsayı değerlerini görüyoruz. Bu değerleri
bağımlı değişken üzerindeki etkilerini yorumlamada kullanabiliriz.

## Model Üzerinden Tahmin
<sup>[İçeriğe dön.](#İçerik)</sup>

Anlamlılığı düşük olduğu için ve önem seviyesi yüksek olmadığı için
***Departure.Delay.in.Minutes*** değişkenini çıkarıyorum.

``` r
trainset<-subset(trainset, select=-c(Departure.Delay.in.Minutes))
```

``` r
predict1<-predict(model1, testset, type="response")
```

``` r
cm1<-InformationValue::confusionMatrix(testset$satisfaction, 
                                      predictedScores = predict1)
cm1
```

    ##   dissatisfied satisfied
    ## 0        12489      4597
    ## 1         2163     22332

Elde edilen confusion matrix incelendiğinde 12489 dissatisfied ve 22332
satisfied değerini doğru tahmin ettiğini (sırasıyla: True Negatives ve
True Positives), 2163 dissadisfied ve 4597 satisfied değerini ise yanlış
tahmin ettiğini (sırasıyla: False Positives ve False Negatives)
görülüyor.

Accuracy(Doğruluk) değeri True Negatives(Doğru Negatif) ve True
Positives(Doğru Pozitif) değerlerinin toplamının tüm tahminlere oranıyla
bulunur.

``` r
accuracy1<-(cm1[2,2]+cm1[1,1])/sum(cm1)
accuracy1
```

    ## [1] 0.8374257

``` r
confmat1<-as.matrix(table(testset$satisfaction, predict1 > 0.5))
confmat1
```

    ##               
    ##                FALSE  TRUE
    ##   dissatisfied 12489  2163
    ##   satisfied     4597 22332

``` r
accuracy0<-(confmat1[1,1]+confmat1[2,2])/sum(confmat1)
accuracy0
```

    ## [1] 0.8374257

Doğru sınıflama yüzdesi yaklaşık %84’tür.

Error Rate(Hata Oranı) ise değeri False Positives(Yanlış Pozitif) ve
False Negatives(Yanlış Negatif) değerlerinin toplamının tüm tahminlere
oranıyla bulunur.

``` r
errorRate1<-(cm1[1,2]+cm1[2,1])/sum(cm1)
```

``` r
errorRate<-(confmat1[1,2]+confmat1[2,1])/sum(confmat1)
errorRate
```

    ## [1] 0.1625743

Hata Oranı yaklaşık %16’dır.

## Optimal Eşik Değeri
<sup>[İçeriğe dön.](#İçerik)</sup>

Tahminler yapılırken en iyi sonucu veren eşik değeri belirlemek için
tahmin özetimizi inceliyoruz.

``` r
summary(predict1)
```

    ##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    ## 0.0001213 0.2037645 0.6736134 0.5752771 0.9205929 0.9961505

Confusion matrix oluştururken elde ettiğimiz değerler için eşik değeri
0.5 olarak belirlenmişti(default). Yani 0.5’in üstü “satisfied
sınıfında, altındaki değerler ise”dissatisfied” sınıfında yer almıştı.
Median değerinin 0.5 üstünde kaldığını görüyoruz. testset’te “satisfied”
verisinin daha fazla bulunmasından da kaynaklı olarak böyle bir sonuç
ortaya çıkmakta.

``` r
optCutoff<-InformationValue::optimalCutoff(testset$satisfaction,
                                           predictedScores=predict1)
optCutoff
```

    ## [1] 0.006150502

Optimal Cut_off değerine baktığımızda çok küçük bir değer aldığını
görüyoruz.

``` r
cmOpt1<-InformationValue::confusionMatrix(testset$satisfaction,
                                         predictedScores = predict1,
                                         threshold=optCutoff)
cmOpt1
```

    ##   dissatisfied satisfied
    ## 0          169        30
    ## 1        14483     26899

Optimal Cut_off değerine göre oluşturduğumuz confusion matrixi
incelediğimizde False Positives değerinin çok büyük olduğunu
görebiliyoruz.

``` r
accuracyopt1<-(cmOpt1[2,2]+cmOpt1[1,1])/sum(cmOpt1)
accuracyopt1
```

    ## [1] 0.6509704

Doğru sınıflama yüzdesi %65 olarak bulundu. Bulunan optimal Cut_off
değeri 0.5 eşik değerinden çok daha kötü bir performans gösterdi.

optimalCutoff fonksiyonu yerine ROCR grafiği üzerinden optimal eşik
değerini bulmaya çalışalım.

``` r
ROCR_pred_test <- prediction(predict1, testset$satisfaction)

ROCR_perf_test <- performance(ROCR_pred_test,'tpr','fpr')
```

![](Çoklu-Lojistik-Regresyon-ile-Havayolu-Müşteri-Memnuniyeti-Analizi_files/figure-gfm/ROCR_perf_test_plot-1.png)<!-- -->

``` r
cost_perf = performance(ROCR_pred_test, "cost")
```

``` r
ROCR_pred_test@cutoffs[[1]][which.min(cost_perf@y.values[[1]])]
```

    ##      3030 
    ## 0.3984497

Elde ettiğimiz yeni eşik değerine göre confusion matrix oluşturalım.

``` r
cmOpt<-InformationValue::confusionMatrix(testset$satisfaction,
                                          predictedScores = predict1,
                                          threshold =0.398 )
cmOpt
```

    ##   dissatisfied satisfied
    ## 0        11544      3449
    ## 1         3108     23480

``` r
cm1
```

    ##   dissatisfied satisfied
    ## 0        12489      4597
    ## 1         2163     22332

Yeni eşik değeri bulduğumuz diğer eşik değerinden çok daha iyi bir
performans göstermekte.0.5 eşik değeri ile kıyaslandığında ise True
Negative mikatrının azaldığını fakat True Positive miktarının arttığı
görülmektedir.

``` r
accuracyopt<-(cmOpt[1,1]+cmOpt[2,2])/sum(cmOpt)
accuracyopt
```

    ## [1] 0.8423078

Doğru sınıflama yüzdesi yaklaşık %84 olarak bulundu.

``` r
accuracy0
```

    ## [1] 0.8374257

0.5 eşik değerinden daha iyi bir performans gösterdi.

Bu veri seti için True Positives mi yoksa True Negatives mi daha önemli
emin değilim. Müşterinin neden memnuniyetsizlik duyabilceğini doğru
tahmin etmek ve ona göre önlem alıp geliştirmek mi yoksa nelerin
memnuniyeti en çok arttırdığını bulup onlara yoğunlaşmak mı? Elimizdeki
çoğu değişken tek kişinin memnuniyetsizliği için değiştirilebilecek
şeyler değil dolayısıyla True Positives’e odaklanmak firma için daha iyi
sonuçlar verebilir.

## Precision ve Recall
<sup>[İçeriğe dön.](#İçerik)</sup>

Kesinlik değeridir. Bir veri setinde ilgilendiğimiz sınıfın aslında ne
kadar doğru sınıflandırıldığını bulmak için kullanılır. İlgilendiğimiz
sınıf pozitifler ise gerçek pozitiflerin sayısının, gerçek pozitifler ve
yanlış pozitiflerin(aslında negatif olan fakat pozitif olarak
sınıflandırılmış durumlar) toplamına oranıyla bulunur.

``` r
precision1<-(cmOpt[2,2]/(cmOpt[2,1]+cmOpt[2,2]))
precision1
```

    ## [1] 0.8831052

``` r
precision<-(confmat1[2,2])/(confmat1[2,1]+confmat1[2,2])
precision
```

    ## [1] 0.8292918

Bulduğumuz eşik değeri, 0.5’ten daha yüksek bir kesinliğe sahiptir.

Bir veri setinde ilgilendiğimiz sınıfın tüm veri noktalarını bulma
yeteneği olarak düşünülebilir. İlgilendiğimiz sınıf pozitifler ise
gerçek pozitiflerin sayısının, gerçek pozitifler ve yanlış
negatiflerin(aslında pozitif olan durumlar) toplamına oranıyla bulunur.

``` r
recall1<-(cmOpt[2,2])/(cmOpt[1,2]+cmOpt[2,2])
recall1
```

    ## [1] 0.8719225

``` r
recall<-(confmat1[2,2])/(confmat1[1,2]+confmat1[2,2])
recall
```

    ## [1] 0.9116963

0.5 eşik değeri ile pozitif değerileri bulma yeteneği seçtiğimiz eşik
değeri ile bulma yeteneğinden daha fazladır.

``` r
recall1_<-(cmOpt[2,2])/(cmOpt[1,1]+cmOpt[2,2])
recall1_
```

    ## [1] 0.6703974

``` r
recall_<-(confmat1[2,2])/(confmat1[1,1]+confmat1[2,2])
recall_
```

    ## [1] 0.6413371

Fakat 0.5 eşik değeri ile negatif değerileri bulma yeteneği seçtiğimiz
eşik değeri ile bulma yeteneğinden daha düşüktür.

**Sensivity ve Specifitiy**

Sensivity(Duyarlılık) recall ile aynı formüle sahiptir.

``` r
sensivity1<-(cmOpt[2,2])/(cmOpt[1,2]+cmOpt[2,2])
sensivity1
```

    ## [1] 0.8719225

``` r
sensivity<-(confmat1[2,2])/(confmat1[1,2]+confmat1[2,2])
sensivity
```

    ## [1] 0.9116963

Negatif sınıfları ne kadar iyi tahmin edebildiğimizi de specificity ile
gorebiliriz.

``` r
specificity1<-(cmOpt[1,1])/(cmOpt[2,1]+cmOpt[1,1])
specificity1
```

    ## [1] 0.7878788

``` r
specificity<-(confmat1[1,1])/(confmat1[2,1]+confmat1[1,1])
specificity
```

    ## [1] 0.7309493

Seçtiğimiz eşik değeri negatif değerleri tahmin etmede daha iyidir.

``` r
f1_score1<-2*((precision1*recall)/(precision1+recall))
f1_score1
```

    ## [1] 0.897173

## F1 Score
<sup>[İçeriğe dön.](#İçerik)</sup>

``` r
f1_score<-2*((precision*recall1)/(precision+recall1))
f1_score
```

    ## [1] 0.850073

Modelin “satisfied” olanları tahmin etme F1 scoru yaklaşık %90 olarak
elde edilmistir.(cut_off=0.389) Modelin placed olanları tahmin etme F1
scoru %85 olarak elde edilmistir.(cut_off=0.5)

## ROC Curve ve AUC (Area Under Curve)
<sup>[İçeriğe dön.](#İçerik)</sup>

    ## Setting levels: control = dissatisfied, case = satisfied

    ## Setting direction: controls < cases

![](Çoklu-Lojistik-Regresyon-ile-Havayolu-Müşteri-Memnuniyeti-Analizi_files/figure-gfm/rocmodel_plot-1.png)<!-- -->

``` r
rocModel
```

    ## 
    ## Call:
    ## roc.formula(formula = testset$satisfaction ~ predict1)
    ## 
    ## Data: predict1 in 14652 controls (testset$satisfaction dissatisfied) < 26929 cases (testset$satisfaction satisfied).
    ## Area under the curve: 0.9115

Modelin altında kalan alan 0.91’dir. Bu değer 1’e yaklaştıkça tahmin
performansı artar. Modelin tahmin performansı gayet iyidir.

## caret Paketi ile Performans Yorumlama
<sup>[İçeriğe dön.](#İçerik)</sup>

``` r
testset$satisfaction <- factor(testset$satisfaction)
```

“dissatisfied” değerleri doğru sınıflandırılmış mı bakalım.

``` r
predictClass<-ifelse(predict1>0.5, "satisfied","dissatisfied")
predictClass<-as.factor(predictClass)
caret::confusionMatrix(predictClass, reference=testset$satisfaction)
```

    ## Confusion Matrix and Statistics
    ## 
    ##               Reference
    ## Prediction     dissatisfied satisfied
    ##   dissatisfied        12489      4597
    ##   satisfied            2163     22332
    ##                                          
    ##                Accuracy : 0.8374         
    ##                  95% CI : (0.8338, 0.841)
    ##     No Information Rate : 0.6476         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.6568         
    ##                                          
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ##                                          
    ##             Sensitivity : 0.8524         
    ##             Specificity : 0.8293         
    ##          Pos Pred Value : 0.7309         
    ##          Neg Pred Value : 0.9117         
    ##              Prevalence : 0.3524         
    ##          Detection Rate : 0.3004         
    ##    Detection Prevalence : 0.4109         
    ##       Balanced Accuracy : 0.8408         
    ##                                          
    ##        'Positive' Class : dissatisfied   
    ## 

``` r
predictClass<-ifelse(predict1>0.398, "satisfied","dissatisfied")
predictClass<-as.factor(predictClass)
caret::confusionMatrix(predictClass, reference=testset$satisfaction)
```

    ## Confusion Matrix and Statistics
    ## 
    ##               Reference
    ## Prediction     dissatisfied satisfied
    ##   dissatisfied        11544      3449
    ##   satisfied            3108     23480
    ##                                           
    ##                Accuracy : 0.8423          
    ##                  95% CI : (0.8388, 0.8458)
    ##     No Information Rate : 0.6476          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6563          
    ##                                           
    ##  Mcnemar's Test P-Value : 2.683e-05       
    ##                                           
    ##             Sensitivity : 0.7879          
    ##             Specificity : 0.8719          
    ##          Pos Pred Value : 0.7700          
    ##          Neg Pred Value : 0.8831          
    ##              Prevalence : 0.3524          
    ##          Detection Rate : 0.2776          
    ##    Detection Prevalence : 0.3606          
    ##       Balanced Accuracy : 0.8299          
    ##                                           
    ##        'Positive' Class : dissatisfied    
    ## 

p_value değerine baktığımızda çok küçük değerde olduğunu görüyoruz.
*dissatsfied*ları doğru sınıflama performansının iyi oldmadığını
söyleyebiliriz.

cut_off = 0.5:  
*dissatisfied*a ait sensitivity değeri 0.85 elde edilmiştir.  
Ayrıca Pos pred value (pozitiflerin doğru tahmin edilme oranı) 0.73 iken
negatiflerin 0.91 görünmektedir. Model *satisfied* olanları daha iyi
tahmin edebilmektedir.  
Balanced accuracy(0.84) de sensivitiy ile specificity bilgisinden elde
edilir ve bunu da yüksek ollması istenen durumdur.

“satisfied” değerleri doğru sınıflandırılmış mı bakalım.

``` r
predictClass<-ifelse(predict1>0.5, "satisfied","dissatisfied")
predictClass<-as.factor(predictClass)
caret::confusionMatrix(predictClass, reference=testset$satisfaction, positive="satisfied")
```

    ## Confusion Matrix and Statistics
    ## 
    ##               Reference
    ## Prediction     dissatisfied satisfied
    ##   dissatisfied        12489      4597
    ##   satisfied            2163     22332
    ##                                          
    ##                Accuracy : 0.8374         
    ##                  95% CI : (0.8338, 0.841)
    ##     No Information Rate : 0.6476         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.6568         
    ##                                          
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ##                                          
    ##             Sensitivity : 0.8293         
    ##             Specificity : 0.8524         
    ##          Pos Pred Value : 0.9117         
    ##          Neg Pred Value : 0.7309         
    ##              Prevalence : 0.6476         
    ##          Detection Rate : 0.5371         
    ##    Detection Prevalence : 0.5891         
    ##       Balanced Accuracy : 0.8408         
    ##                                          
    ##        'Positive' Class : satisfied      
    ## 

``` r
predictClass<-ifelse(predict1>0.398, "satisfied","dissatisfied")
predictClass<-as.factor(predictClass)
caret::confusionMatrix(predictClass, reference=testset$satisfaction, positive="satisfied")
```

    ## Confusion Matrix and Statistics
    ## 
    ##               Reference
    ## Prediction     dissatisfied satisfied
    ##   dissatisfied        11544      3449
    ##   satisfied            3108     23480
    ##                                           
    ##                Accuracy : 0.8423          
    ##                  95% CI : (0.8388, 0.8458)
    ##     No Information Rate : 0.6476          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6563          
    ##                                           
    ##  Mcnemar's Test P-Value : 2.683e-05       
    ##                                           
    ##             Sensitivity : 0.8719          
    ##             Specificity : 0.7879          
    ##          Pos Pred Value : 0.8831          
    ##          Neg Pred Value : 0.7700          
    ##              Prevalence : 0.6476          
    ##          Detection Rate : 0.5647          
    ##    Detection Prevalence : 0.6394          
    ##       Balanced Accuracy : 0.8299          
    ##                                           
    ##        'Positive' Class : satisfied       
    ## 

cut_off = 0.5:  
*satisfied*a ait sensivitiy değeri yaklaşık 0.83 elde edilmiştir. Ayrıca
Pos pred value (pozitiflerin doğru tahmin edilme oranı) 0.91 iken
negatiflerin 0.73 görünmektedir. Model *satisfied* olanları daha iyi
tahmin edebilmektedir.  
Balanced accuracy(0.84) de sensivitiy ile specificity bilgisinden elde
edilir ve bunu da yüksek olması istenen durumdur.  
———————————————————————————————————————————————————————-

``` r
predictClass<-ifelse(predict1>0.5, "satisfied","dissatisfied")
predictClass<-as.factor(predictClass)
caret::confusionMatrix(predictClass, reference=testset$satisfaction, positive="satisfied", mode="prec_recall")
```

    ## Confusion Matrix and Statistics
    ## 
    ##               Reference
    ## Prediction     dissatisfied satisfied
    ##   dissatisfied        12489      4597
    ##   satisfied            2163     22332
    ##                                          
    ##                Accuracy : 0.8374         
    ##                  95% CI : (0.8338, 0.841)
    ##     No Information Rate : 0.6476         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.6568         
    ##                                          
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ##                                          
    ##               Precision : 0.9117         
    ##                  Recall : 0.8293         
    ##                      F1 : 0.8685         
    ##              Prevalence : 0.6476         
    ##          Detection Rate : 0.5371         
    ##    Detection Prevalence : 0.5891         
    ##       Balanced Accuracy : 0.8408         
    ##                                          
    ##        'Positive' Class : satisfied      
    ## 

``` r
predictClass<-ifelse(predict1>0.398, "satisfied","dissatisfied")
predictClass<-as.factor(predictClass)
caret::confusionMatrix(predictClass, reference=testset$satisfaction, positive="satisfied", mode="prec_recall")
```

    ## Confusion Matrix and Statistics
    ## 
    ##               Reference
    ## Prediction     dissatisfied satisfied
    ##   dissatisfied        11544      3449
    ##   satisfied            3108     23480
    ##                                           
    ##                Accuracy : 0.8423          
    ##                  95% CI : (0.8388, 0.8458)
    ##     No Information Rate : 0.6476          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6563          
    ##                                           
    ##  Mcnemar's Test P-Value : 2.683e-05       
    ##                                           
    ##               Precision : 0.8831          
    ##                  Recall : 0.8719          
    ##                      F1 : 0.8775          
    ##              Prevalence : 0.6476          
    ##          Detection Rate : 0.5647          
    ##    Detection Prevalence : 0.6394          
    ##       Balanced Accuracy : 0.8299          
    ##                                           
    ##        'Positive' Class : satisfied       
    ## 

cut_off = 0.398 için Accuracy, Recall, F1, Detection Rate ve Detection
Prevalence değerleri; cut_off = 0.5’ten daha yüksektir. Precision değeri
ise cut_off = 0.5 için daha yüksektir.  
cut_off = 0.398 iken memnun olan müşterilerin tespit edilme oranı daha
yüksektir. Memnun olarak tespit edilen müşterilerin doğru
sınıflandırılma oranı ise cut_off = 0.5 iken daha yüksektir.  
Kappa değeri karşılaştırmalı uyuşmanın güvenirliğini gösterir. 0.65
değeri iyidir.

İki cut_off noktasının presicion değerleri arasında çok büyük bir fark
yoktur. Müşteri memnuniyeti presicionın öneminin yüksek olduğu bir durum
değildir. Genel anlamda daha olumlu değerlere sahip olduğu için cut_off
= 0.398 seçilebilir.
