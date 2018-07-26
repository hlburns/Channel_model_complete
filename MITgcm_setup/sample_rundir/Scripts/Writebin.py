class binfiles:
    def __init__(self):
        self.buffersize=8192

    def writeOrg(self,filename,matrix,type='d'):
        # create an array first
        aMatrix=A.array(type)
        # turn matrix into a flat and then into a list
        # the list is then read into aMatrix
        aMatrix.fromlist(matrix.flat.tolist())
        # open the file
        fd=open(filename,'wb')
        aMatrix.tofile(fd)
        fd.close()

    def write(self,filename,matrix):
        # open the file
        fd=open(filename,'wb')
        matrix.ravel().tofile(fd)
        fd.close()

    def read(self,filename,size=(0,0)):
        fd=open(filename,'rb')
        buffersize=size[0]*size[1]
        if (buffersize==0) or (buffersize>self.buffersize):
            buffersize=self.buffersize
            
        while True:
            try:
                matrix.fromfile(fd,buffersize)
            except EOFError:
                break
            
        fd.close()
        if size[0]*size[1]!=0:
            # size is given
            matrix=reshape(matrix,size)
        return(matrix)
def writeFile(filename,X):
    bf=binfiles()
    bf.write(filename,X.astype('float64'))
