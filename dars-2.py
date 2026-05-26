# Keling, juda sodda, kirish xususiyatlarini qabul qilib, ularni qayta ishlaydigan neyron tarmog‘ini yozamiz.

import torch
import torch.nn as nn # nn - Neural Networks (Neyron tarmoqlari) moduli

# 1. Shaxsiy model klasimizni yaratamiz
class KichikModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Kirish qatlami: 4 ta raqam qabul qiladi (Linear / Dense qatlam)
        # Chiqish qatlami: 2 ta raqam qaytaradi
        # Masalan: [bo'yi, vazni, yoshi, jinsi] -> [Sog'lom, Kasal] ehtimolligi
        self.qatlam = nn.Linear(in_features=4, out_features=2)
        
    def forward(self, x):
        # Ma'lumotni qatlamdan o'tkazamiz
        chiqish = self.qatlam(x)
        return chiqish

# 2. Model obyektini yaratamiz
model = KichikModel()
print("Model Arxitekturasi:\n", model)

# 3. Modelga berish uchun sun'iy ma'lumot (Tensor) tayyorlaymiz
# Aytaylik, bizda 3 ta odam haqida ma'lumot bor (Batch size = 3), har birida 4 tadan xususiyat
faydalanuvchilar = torch.tensor([
    [175.0, 70.0, 25.0, 1.0],  # 1-odam
    [160.0, 55.0, 30.0, 0.0],  # 2-odam
    [185.0, 90.0, 45.0, 1.0]   # 3-odam
])

print("\nKiruvchi ma'lumot shakli:", faydalanuvchilar.shape) # torch.Size([3, 4])

# 4. Modelni ishga tushiramiz (Forward pass)
# Diqqat: model.forward(faydalanuvchilar) deb yozish ham mumkin, 
# lekin PyTorch-da obyektni to'g'ridan-to'g'ri chaqirish (model(x)) tavsiya etiladi.
bashorat = model(faydalanuvchilar)

print("\nModel bergan javob (Bashorat):\n", bashorat)
print("Chiquvchi ma'lumot shakli:", bashorat.shape) # torch.Size([3, 2])