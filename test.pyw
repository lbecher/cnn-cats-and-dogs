from tkinter import Button, Canvas, filedialog, Label, Menu, messagebox, NW, Scale, Tk, Toplevel
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk


class App:
    def __init__(self, raiz):
        # Parâmetros iniciais
        self.imagem = None
        self.modelo = None

        self.raiz = raiz
        self.raiz.title("Reconhecimento de gatos e cachorros")

        # Barra de menu
        self.barra_de_menu = Menu(self.raiz)
        self.menu_de_arquivo = Menu(self.barra_de_menu, tearoff=0)
        self.menu_de_arquivo.add_command(label="Abrir imagem...", command=self.abrir_imagem)
        self.menu_de_arquivo.add_command(label="Abrir modelo...", command=self.abrir_modelo)
        self.menu_de_arquivo.add_separator()
        self.menu_de_arquivo.add_command(label="Fechar", command=self.fechar_aplicativo)
        self.barra_de_menu.add_cascade(label="Arquivo", menu=self.menu_de_arquivo)

        self.menu_de_ferramentas = Menu(self.barra_de_menu, tearoff=0)
        self.menu_de_ferramentas.add_command(label="Classificar imagem", command=self.classificar_imagem)
        self.barra_de_menu.add_cascade(label="Ferramentas", menu=self.menu_de_ferramentas)

        self.raiz.config(menu=self.barra_de_menu)

        # Exibição da imagem
        self.canvas = Canvas(self.raiz, width=800, height=600)
        self.canvas.pack(padx=10, pady=10)
    
    def abrir_imagem(self):
        caminho = filedialog.askopenfilename(defaultextension=".png", filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("Todos os arquivos", "*.*")])
        if caminho:
            # Abre a imagem
            self.imagem = caminho
            self.mostrar_imagem()
    
    def abrir_modelo(self):
        caminho = filedialog.askopenfilename(defaultextension=".keras", filetypes=[("Keras", "*.keras"), ("Todos os arquivos", "*.*")])
        if caminho:
            # Abre o modelo
            self.modelo = tf.keras.models.load_model(caminho)
    
    def fechar_aplicativo(self):
        self.raiz.quit()
    
    def mostrar_imagem(self):
        if self.imagem is not None:
            imagem = Image.open(self.imagem)

            width, height = imagem.size
            max = 800
            if width > max or height > max:
                ratio = min(max / width, max / height)
                width = int(width * ratio)
                height = int(height * ratio)
                imagem = imagem.resize((width, height))
            
            self.imagem_de_exibicao = ImageTk.PhotoImage(imagem)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=NW, image=self.imagem_de_exibicao)
        else:
            messagebox.showerror("Erro", "Nenhuma imagem foi aberta!")
    
    def processar_imagem(self):
        imagem = image.load_img(self.imagem, target_size=(192, 192), color_mode='grayscale')
        imagem_array = image.img_to_array(imagem)
        imagem_array = np.expand_dims(imagem_array, axis=0)  # Adiciona uma dimensão extra para o lote
        return imagem_array
    
    def classificar_imagem(self):
        if self.imagem is not None and self.modelo is not None:
            # Processa a imagem para o reconhecimento
            imagem_processada = self.processar_imagem()

            # Classificação usando o modelo carregado
            predicoes = self.modelo.predict(imagem_processada)
            classe = np.argmax(predicoes)  # Obtém a classe prevista (0 para gato, 1 para cachorro)
            confianca = round(np.max(predicoes) * 100.0)  # Obtém a confiança (probabilidade) da classe prevista

            classes = ["gato", "cachorro"]

            messagebox.showinfo("Reconhecimento", "Isso é um " + classes[classe] + ", confia! Tenho " + str(confianca) + "% de certeza!")
        else:
            messagebox.showerror("Erro", "Nenhuma imagem e/ou nenhum modelo foi aberto!")


if __name__ == "__main__":
    raiz = Tk()
    app = App(raiz)
    raiz.mainloop()


"""
# Carregar o modelo salvo
model_path = 'gatos_e_cachorros_25.keras'
loaded_model = tf.keras.models.load_model(model_path)

# Função para carregar e redimensionar uma imagem para o tamanho esperado pelo modelo
def load_and_preprocess_image(image_path, target_size=(192, 192)):
    img = image.load_img(image_path, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão extra para o lote
    return img_array

# Caminho da imagem a ser classificada (substitua pelo caminho da sua imagem)
image_path = 'caminho/para/sua/imagem.jpg'

# Carregar e pré-processar a imagem
input_image = load_and_preprocess_image(image_path)

# Classificação usando o modelo carregado
predictions = loaded_model.predict(input_image)
predicted_class = np.argmax(predictions)  # Obtém a classe prevista (0 para gato, 1 para cachorro)

# Mapa de classes para rótulos
class_names = ['gato', 'cachorro']

# Resultado da classificação
predicted_label = class_names[predicted_class]
print("A imagem é classificada como:", predicted_label)
"""