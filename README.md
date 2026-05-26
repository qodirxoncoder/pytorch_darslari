
# 🧠 PyTorch va LLM asoslari (Learning Journey)

Ushbu repozitoriyada Katta til modellari (LLM) qanday ishlashi va ularning ortidagi arxitekturani noldan tushunish uchun PyTorch-da yozilgan amaliy darslar va kodlar jamlangan. Maqsad — sun'iy intellekt modellarini tayyor API-lardan foydalanib emas, ichki mantiq va matematika (Transformer, Attention, Embedding) orqali noldan o'rganish.

---

## 🚀 Loyiha tuzilishi (Darslar)

| Fayl nomi | Mavzu | Kontent / O'rganilgan tushunchalar | Status |
| :--- | :--- | :--- | :--- |
| `dars-1.py` | **PyTorch Asoslari** | Tensorlar, Shape (o'lchamlar), Matritsalar ko'paytmasi (`@`) va GPU (CUDA) integratsiyasi. | ✅ Bajarildi |
| `dars-3.py` | **Training Loop (O'qitish sikli)** | `nn.Module`, Forward/Backward pass, Loss funksiyasi va Gradientlar portlashi (*Exploding Gradients*) muammosini normalizatsiya orqali hal qilish. | ✅ Bajarildi |
| `dars-4.py` | **NLP va Embedding** | Matnni tokenizatsiya qilish, unikal Lug'at (*Vocabulary*) tuzish va so'zlarni ma'noli ko'p o'lchamli vektorlarga (`nn.Embedding`) o'girish. | ✅ Bajarildi |
| *Tez orada...* | **Attention (Diqqat) mexanizmi** | Transformer arxitekturasining yuragi bo'lgan Self-Attention mexanizmini noldan yozish. | ⏳ Rejada |

---

## 🛠 O'rnatish va Ishga tushirish

Loyiha **Python 3.14** (yoki undan yuqori) hamda PyTorch virtual muhitida ishlaydi.

1. **Repozitoriyani yuklab oling:**
   ```bash
   git clone [https://github.com/sizingizning_username/pytorch_darslari.git](https://github.com/sizingizning_username/pytorch_darslari.git)
   cd pytorch_darslari
