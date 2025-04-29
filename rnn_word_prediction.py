
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# النص المسستخدم 4 كلمات
text = "khaled alnagdi mohamed alsayed"

# تحويل النصص إلى أرقام
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequence = tokenizer.texts_to_sequences([text])[0]

# تجهيز البيانات المدخلات  أول 3 كلمات الهدف = الكلمة الرابعة
x = np.array([sequence[:3]])  
y = to_categorical(sequence[3], num_classes=len(tokenizer.word_index) + 1)

# بناء نموذج RNN
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10)) 
model.add(SimpleRNN(16))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

# تدريبب النموذج
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=500, verbose=0)

# اختبار التنبؤ بالكلمة الرابعة
pred = model.predict(x, verbose=0)
predicted_index = pred.argmax()
predicted_word = tokenizer.index_word[predicted_index]

print("Predicted word:", predicted_word)
