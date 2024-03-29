Reconhecimento Facial com face_recognition

Este é um projeto de reconhecimento facial desenvolvido utilizando a biblioteca face_recognition em Python. Esta aplicação tem como objetivo identificar e reconhecer rostos em imagens o, seguindo os passos descritos abaixo:

Passos do Processo de Reconhecimento Facial

Detecção de Rosto: Utiliza algoritmos de detecção de rosto para localizar e extrair as características faciais presentes em uma imagem ou vídeo.

Extração de Características: Após a detecção do rosto, extrai características únicas, como a posição dos olhos, nariz, boca e outros traços distintivos.

Comparação com Banco de Dados: As características são comparadas com um banco de dados de rostos previamente cadastrados, utilizando técnicas de aprendizado de máquina para calcular a similaridade entre os rostos.

Identificação ou Verificação: Dependendo do contexto, o sistema pode realizar duas tarefas principais:
    1)Identificação: Tenta determinar a identidade do rosto desconhecido comparando-o com todos os rostos no banco de dados.
    2)Verificação: Verifica se o rosto desconhecido corresponde a uma pessoa específica previamente cadastrada.

Retorno de Resultados: Com base na comparação realizada, o sistema retorna os resultados, como a identidade da pessoa reconhecida ou a probabilidade de correspondência com uma pessoa específica.

Considerações Importantes

O sucesso do reconhecimento facial depende de fatores como a qualidade das imagens, a precisão dos algoritmos de detecção e reconhecimento, e a quantidade e qualidade dos dados de treinamento disponíveis.
Questões éticas e de privacidade devem ser consideradas ao implementar sistemas de reconhecimento facial, especialmente em relação ao armazenamento e uso dos dados biométricos das pessoas.

Atenção! Para que o código funcione localmente é necessário utilizar o sistema operacional Linux ou MacOS. Além disso, é imprescindivel a criação das pastas 'images' e 'dados'. 
    
Para o código base de reconhecimento no Google Colab é importante selecionar o tipo de ambiente de execução 'GPU' para evitar erros. 

Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para relatar problemas. 
