# Modelni o‘qitish mexanizmi (Uchburchak asosi)

import torch
import torch.nn as nn
import torch.optim as optim  # Optimallashtirish modullari

class KichikModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qatlam = nn.Linear(4, 2)
        
    def forward(self, x):
        return self.qatlam(x)

model = KichikModel()

# 1. Sun'iy kirish ma'lumotlari (X)
# 1. Sun'iy kirish ma'lumotlarini normalizatsiya qilamiz (kichik sonlarga keltiramiz)
# Bo'yi, vazni, yoshi kabi qiymatlarni kichraytirdik
faydalanuvchilar = torch.tensor([
    [1.75, 0.70, 0.25, 1.0],  # 175.0 -> 1.75 va hokazo
    [1.60, 0.55, 0.30, 0.0],
    [1.85, 0.90, 0.45, 1.0]
])

# 2. Haqiqiy to'g'ri javoblar o'sha-o'shaligicha qoladi
haqiqiy_javoblar = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0]
])

# 3. Optimizer tezligini biroz oshiramiz (chunki sonlar kichraydi)
optimizer = optim.SGD(model.parameters(), lr=0.1) # lr ni 0.1 qildik

# 2. Haqiqiy to'g'ri javoblar (Y target) - Model aynan shu javoblarga intilishi kerak
# Aytaylik, birinchi va uchinchi odam [1, 0] (sog'lom), ikkinchi odam [0, 1] (kasal)
haqiqiy_javoblar = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0]
])

# 3. Xatolik funksiyasi va Optimizer-ni belgilaymiz
loss_fn = nn.MSELoss()  # Mean Squared Error (O'rtacha kvadratik xatolik)
optimizer = optim.SGD(model.parameters(), lr=0.1)  # lr - Learning Rate (o'rganish tezligi)

print("O'qitish boshlanishidan oldingi ilk bashorat:\n", model(faydalanuvchilar))
print("-" * 50)

# 4. O'QITISH CYCLE (TRAINING LOOP)
# Modelni 100 marta qayta-qayta o'qitamiz
for epoch in range(100):
    # a) Forward pass: Modelda bashorat qilamiz
    bashorat = model(faydalanuvchilar)
    
    # b) Xatolikni hisoblaymiz
    loss = loss_fn(bashorat, haqiqiy_javoblar)
    
    # c) Gradientlarni tozalaymiz (PyTorch har safar gradientlarni qo'shib ketmasligi uchun)
    optimizer.zero_grad()
    
    # d) Backward pass: Xatolikni orqaga qarab tarqatamiz (Hosilalar hisoblanadi)
    loss.backward()
    
    # e) Vaznlarni yangilaymiz (Optimizer qadam tashlaydi)
    optimizer.step()
    
    # Har 20 ta qadamda xatolik kamayayotganini tekshiramiz
    if (epoch + 1) % 20 == 0:
        print(f"Epoxa [{epoch+1}/100], Xatolik (Loss): {loss.item():.4f}")

print("-" * 50)
print("O'qitishdan keyingi yakuniy bashorat:\n", model(faydalanuvchilar))
print("Aslida bo'lishi kerak bo'lgan javoblar:\n", haqiqiy_javoblar)