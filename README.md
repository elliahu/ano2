1. Vytvoření virtuálního prostředí + aktivace + :
	python3 -m venv env
	source env/bin/activate
    pip3 install --upgrade pip

2. Instalace potřebných knihoven:
	pip3 install -r requirements.txt

3. Spuštění ukázky:
	python3 main.py

# Porovnání modelů
## Custom models
- **Custom Pytorch CNN**: 72.4702380952381%
- **HAAR**: 63.39285714285714%
- **Edge detection & thresholding**: 87.42559523809524%
## GoogLeNet (GoogLeNetSmall)
![googlenet_model.pth.png](models/googlenet_model.pth.png)
```py
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

|Image |Success rate [%]|
|-----|------------|
test_images\test1.jpg|100.0
test_images\test10.jpg|98.21428571428571
test_images\test11.jpg|96.42857142857143
test_images\test12.jpg|100.0
test_images\test13.jpg|91.07142857142857
test_images\test14.jpg|89.28571428571429
test_images\test15.jpg|76.78571428571429
test_images\test16.jpg|100.0
test_images\test17.jpg|87.5
test_images\test18.jpg|17.857142857142858
test_images\test19.jpg|66.07142857142857
test_images\test2.jpg|98.21428571428571
test_images\test20.jpg|64.28571428571429
test_images\test21.jpg|66.07142857142857
test_images\test22.jpg|16.071428571428573
test_images\test23.jpg|21.428571428571427
test_images\test24.jpg|94.64285714285714
test_images\test3.jpg|96.42857142857143
test_images\test4.jpg|98.21428571428571
test_images\test5.jpg|100.0
test_images\test6.jpg|98.21428571428571
test_images\test7.jpg|89.28571428571429
test_images\test8.jpg|91.07142857142857
test_images\test9.jpg|87.5

**Average success rate**: 81.02678571428572 %

I was surprised to see that the success rate was in fact lower than prehistoric method of counting edge pixels. 

## ResNet18
```py
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
|Image |Success rate [%]|
|-----|------------|
test_images\test1.jpg|78.57142857142857
test_images\test10.jpg|62.5
test_images\test11.jpg|92.85714285714286
test_images\test12.jpg|73.21428571428571
test_images\test13.jpg|8.928571428571429
test_images\test14.jpg|1.7857142857142856
test_images\test15.jpg|3.571428571428571
test_images\test16.jpg|50.0
test_images\test17.jpg|100.0
test_images\test18.jpg|1.7857142857142856
test_images\test19.jpg|3.571428571428571
test_images\test2.jpg|85.71428571428571
test_images\test20.jpg|3.571428571428571
test_images\test21.jpg|3.571428571428571
test_images\test22.jpg|3.571428571428571
test_images\test23.jpg|3.571428571428571
test_images\test24.jpg|8.928571428571429
test_images\test3.jpg|75.0
test_images\test4.jpg|89.28571428571429
test_images\test5.jpg|83.92857142857143
test_images\test6.jpg|92.85714285714286
test_images\test7.jpg|96.42857142857143
test_images\test8.jpg|71.42857142857143
test_images\test9.jpg|76.78571428571429

**Average success rate**: 48.8095238095238 %

From the start, I noticed that the model takes considerable amount of time (more than a second up to few seconds) to produce its prediction. The success rate is not very good in comparison with GoogLeNet. It produces mostly correct results 75% + in good weather sunny pictures, but for pictures in night, fog or generaly not in good condition, it produces mostly wrong (sometimes all together wrong) predictions.