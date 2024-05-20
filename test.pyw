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
        self.menu_de_arquivo.add_command(label="(1º) Abrir modelo...", command=self.abrir_modelo)
        self.menu_de_arquivo.add_command(label="(2º) Abrir imagem...", command=self.abrir_imagem)
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
        imagem = image.load_img(self.imagem, target_size=(128, 128), color_mode='grayscale')
        imagem_array = image.img_to_array(imagem)
        imagem_array = np.expand_dims(imagem_array, axis=0)
        imagem_array = imagem_array / 255.0
        return imagem_array
    
    def classificar_imagem(self):
        if self.imagem is not None and self.modelo is not None:
            # Processa a imagem para o reconhecimento
            imagem_processada = self.processar_imagem()

            # Classificação usando o modelo carregado
            predicoes = self.modelo.predict(imagem_processada)
            classe = int(np.argmax(predicoes, axis=-1))
            porcentagens = predicoes[0]
            porcentagens = porcentagens * 100

            classes = ["gato", "cachorro"]

            mensagem = "Isso é um " + classes[classe] + ", confia! Tenho certeza absoluta!\n\n"
            mensagem += f"Gato: {porcentagens[0]:.2f}%\nCachorro: {porcentagens[1]:.2f}%"
            messagebox.showinfo("Reconhecimento", mensagem)          
        else:
            messagebox.showerror("Erro", "Nenhuma imagem e/ou nenhum modelo foi aberto!")


if __name__ == "__main__":
    raiz = Tk()
    app = App(raiz)
    raiz.mainloop()
