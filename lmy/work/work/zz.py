# -*- coding: utf-8 -*- #文件也为UTF-8
from PIL import Image ,ImageDraw,ImageFont
def read(): 
    i=0
    f = open("shequ.txt",'r')
    for line in f:
        single_line = line.split('\t')
        if(len(single_line)>10):
            i=i+1
            print  "process " +str(i)+ " user "


            img = Image.open("a.png") 
            font = ImageFont.truetype('simsun.ttc',40)#.ttf  simsun
            draw = ImageDraw.Draw(img)
            draw.text((400,250),unicode(single_line[1],'utf-8'),(255,0,0),font=font)
            draw.text((300,300),unicode( "个人社区责履行报告",'utf-8'),(255,0,0),font=font) 
            img.save(str(i)+"a"+".png",'PNG')


            img = Image.open("b.png") 
            font = ImageFont.truetype('simsun.ttc',40)#.ttf  simsun
            draw = ImageDraw.Draw(img)
            draw.text((150,150),unicode(single_line[2],'utf-8'),(255,0,0),font=font)
            draw.text((494,470),unicode(single_line[4],'utf-8'),(255,0,0),font=font)
            draw.text((700,620),unicode(single_line[3],'utf-8'),(255,0,0),font=font)
            img.save(str(i)+"b"+".png",'PNG')



            img = Image.open("c.png") 
            font = ImageFont.truetype('simsun.ttc',40)#.ttf  simsun
            draw = ImageDraw.Draw(img)
            draw.text((150,550),unicode(single_line[12],'utf-8'),(255,0,0),font=font)
            draw.text((750,550),unicode(single_line[12],'utf-8'),(255,0,0),font=font)
            img.save(str(i)+"c"+".png",'PNG')


            img = Image.open("d.png") 
            img.save(str(i)+"d"+".png",'PNG')


            img = Image.open("e.png") 
            font = ImageFont.truetype('simsun.ttc',40)#.ttf  simsun
            draw = ImageDraw.Draw(img)
            draw.text((420,215),unicode(single_line[5],'utf-8'),(255,0,0),font=font)
            draw.text((780,205),unicode(single_line[6],'utf-8'),(255,0,0),font=font)
            draw.text((315,390),unicode(single_line[7],'utf-8'),(255,0,0),font=font)
            draw.text((590,390),unicode(single_line[8],'utf-8'),(255,0,0),font=font)
            draw.text((865,390),unicode(single_line[9],'utf-8'),(255,0,0),font=font)
            draw.text((510,590),unicode(single_line[10],'utf-8'),(255,0,0),font=font)
            img.save(str(i)+"e"+".png",'PNG')
    f.close()

   
if __name__ == '__main__':
    read()
