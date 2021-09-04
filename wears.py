class wearr:

    def __init__(self,gray_image):
        self.gray_image=gray_image
        
    def read(self,path):
        '''__________________________read________________________________'''
        img=cv2.imread(path,0)
        return img
     
    def contour_find(self,img,level):
        '''__________________________skimage countor find________________________________'''
        #gray=skimage.color.rgb2gray(img)
        from skimage import filters
        import numpy as np
        import skimage
        import matplotlib.image
        import skimage.color
        # img=matplotlib.image.imread(path)
        # grey=skimage.color.rgb2grey(img)
        img_edges=filters.sobel(img)
        import skimage.measure
        contours=skimage.measure.find_contours(img_edges,level,fully_connected='low',positive_orientation='high')
        contours=np.array(contours)
        return contours
    
    def contour_show(self,countours):
        import numpy as np

        fig,ax=plt.subplots()
        ax.imshow(self.gray_image, interpolation='nearest', cmap=plt.cm.gray)
        for n, contour in enumerate(countours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
                 
    def bad_row(self,contours):
        import numpy as np
        bad_row=[]
        for i in np.arange(0,len(contours)):
            if len(contours[i]) <=400:
                bad_row.append(i)
        aaa=np.delete(contours,bad_row,axis=0)
        return aaa
    
    def xy_expand(self,x,y):
        import numpy as np
        xmin,xmax,ymin,ymax=min(x),max(x),min(y),max(y)
        width=xmax-xmin
        height=ymax-ymin
        cx=(int(width/2)+int(xmin))
        cy=(int(height/2)+int(ymin))
        R=int(width/2)
        return xmin,xmax,ymin,ymax,width,height,cx,cy,R
    
    
    def contour_cv2(self,img_gray):
        import cv2
        ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        show=cv2.drawContours(img_gray, contours, -1, (0,255,0), 2)
        cnt = contours[0]
        return contours,show
    
    
    def area_find(self,contours):
        import cv2
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        return area
    def contour_xy(self,contour,N):
        import numpy as np
        n=contour.shape
        xxx=[]
        yyy=[]
        for i in np.arange(0,len(contour[N])):
            a=contour[N][i][1]
            xxx.append(a)
        for i in np.arange(0,len(contour[N])):
            a=contour[N][i][0]
            yyy.append(a) 
        xxx=np.array(xxx)
        yyy=np.array(yyy)
        return xxx,yyy
    
    
    
    #   
    def find_wear(self,idd,level):
        '''
        _
        +
        _
        =
        find wear of cutter 
        '''
        from scipy.signal import convolve2d
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        import matplotlib.transforms as transforms
        #--------------------------------------------
        #-----------------------------PREPROCESSING--
        #--------------------------------------------
        #------------READ--------------------------------
        gray=self.gray_image
        ho,wo=gray.shape
        Hx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        Hy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)
        Gx = convolve2d(gray, Hx)
        Gy = convolve2d(gray, Hy)
        G = np.sqrt(Gx*Gx + Gy*Gy)
        G=cv2.resize(G,(wo,ho))
        SHARPY = np.array([[-1, -1, -1],
                        [-1, 20, -1],
                        [-1, -1, -1]])
        sharp=cv2.filter2D(gray,-1,SHARPY)
        #-----------------------------SATART CONTOUR-----
        #------------------------------------------------
        oimage=G
        org_counter=self.contour_find(gray,level)      #FIND CONTOURS WITH SKIMAGE
        org_counter=self.bad_row(org_counter)        # REMOVE BAD ROWS FROM CONTOURS 
        number_of_instance=org_counter.shape[0]
        
        xx,yy=self.contour_xy(org_counter,idd)       #GET X & Y FROM CONTOUR
        xmino,xmaxo,ymino,ymaxo,widtho,heighto,cxo,cyo,Ro=self.xy_expand(xx,yy)
        #------------DRAW--------------------------------
        #--------------------&---------------------------
        #-----------------------SAVE---------------------
        #plt.figure()
        
        
        fig,ax=plt.subplots(nrows=1,ncols=1)
        
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(180)
        
        ax.plot(-xx,yy,',r--',linewidth=5,transform= rot + base)

        
        #a=plt.plot(-xx,yy,',r--',linewidth=5,transform= rot + base),plt.title('{}'.format(level)),plt.axis('off')
        plt.savefig('plot.jpg')
        
        plt.close(fig)
        
        contour_image=cv2.imread('plot.jpg',0)
        contour_image=cv2.resize(contour_image,(int(widtho*0.88),int(heighto*0.88)))                 # * * * * * * * 
        hc,wc=contour_image.shape

        # find_contour OF contour_image
        contour_image_contour=self.contour_find(contour_image,0.1)
        # get data from contour OF contour_image
        x,y=self.contour_xy(contour_image_contour,0)
        xmin,xmax,ymin,ymax,width,height,cx,cy,R=self.xy_expand(x,y)

        contours,draw_contour_org=self.contour_cv2(contour_image)
        area=self.area_find(contours)
        #sh(draw_contour_org)
        #ee=cv2.circle(qq,(cx,cy),radius=R,color=(0,0,255))
        #------------DRAW--------------------------------
        #--------------------a---------------------------
        #-----------------------circle---------------------
        circle=(np.zeros((ho,wo),dtype='float32'))
        circle=cv2.resize(circle,(wo,ho))
        circle=cv2.circle(circle,(cxo,cyo),radius=Ro,color=(255,100,255),thickness=12,lineType=cv2.LINE_AA)
        circle=circle.astype(np.uint8)
        circle_contours,show_circle=self.contour_cv2(circle)
        intact_area=self.area_find(circle_contours)
        percent_of_wear=area/intact_area
        #--------------------------------------------
        #======================show====
        #--------------------------------------------
        #contour_show(oimage,org_counter)          # SHOW CONTOUR 

        #--------------------------------------------
        #======================print informations====
        #--------------------------------------------

        #print('xmin {} \nxmax {} \nymin {} \nymax {} \n'.format(xmin,xmax,ymin,ymax))                                   
        #print('width {} \nheight {} \nR {} \ncenter {}  \nintact_area: {} '.format(width,height,
         #                                                                   R,(cx,cy),intact_area))
        #print('percent_of_wear: ',percent_of_wear)
        return intact_area,  area   ,percent_of_wear,show_circle   ,draw_contour_org,    number_of_instance  
    
    
    def num_instance(self,contours):
        N=contours.shape
        return N
    
    def level(self):
        import numpy as np
        n_ins=[]
        for i in np.arange(1,100):
            level_contourfind=0.003*i
            a=self.contour_find(self.gray_image,level_contourfind)
            b=self.bad_row(a)
            c=self.num_instance(b)
            if c[0] > 0:
                n_ins.append([c,level_contourfind])
        n_ins=np.array(n_ins)        
        return n_ins 
    def best_value(self):
        import numpy as np
        a=self.level()
        value=[]
        for i in np.arange(0,len(a)):
            b=a[:,0][i][0]-1
            c=a[:,1][i]
            value.append([b,c])
        value=np.array(value)
        percent_value=[]
        for i in np.arange(0,len(a)):
            intact_area,  area   ,percent_of_wear,show_circle   ,draw_contour_org,    number_of_instance =self.find_wear(int(a[:,1][i]),a[:,1][i])
            percent_value100=percent_of_wear*100
            percent_value100=np.asarray(percent_value100,dtype='int16')
            if percent_value100 < 100 :
                percent_value.append(percent_of_wear)
            #print('percent_of_wear={}|instance num={}|level={}'.format(percent_of_wear,int(level[:,1][i]),level[:,1][i]))
        

        return percent_value
        
    def ss(self,gray):
        Hx = np.array([[2,1,0,-1,-2],
                       [2,1,0,-1,-2],
                       [4,2,0,-2,-4],
                       [2,1,0,-1,-2],
                       [2,1,0,-1,-2]],dtype=np.float32)
        
        
        Hy = np.array([[2,2,4,2,2],
                       [1,1,2,1,1],
                       [0,0,0,0,0],
                       [-1,-1,-2,-1,-1],
                       [-2,-2,-4,-2,-2]],dtype=np.float32)
        Gx = convolve2d(gray, Hx)
        Gy = convolve2d(gray, Hy)
        G = np.sqrt(Gx*Gx + Gy*Gy)
        return G
    def filtering(self,img,kx,lt):
        img=np.asarray(img,dtype=np.float)
        img=img*kx+lt
        img[img>255]=255
        img[img<0]=0
        return img

