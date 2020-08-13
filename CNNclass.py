import numpy as np

class Conv :
    def __init__ ( self , num ) :
        self.num = num

        self.filters = np.random.randn ( num , 3 , 3 ) / 9

    def iter_region ( self , img ) :
        h , w = img.shape

        for i in range ( h - 2 ) :
            for j in range ( w - 2 ) :
                region = img [ i : i + 3 , j : j + 3 ]

                yield region , i , j

    def forward ( self , input ) :
        h , w = input.shape

        output = np.zeros ( h - 2 , w - 2 , self.num )

        for region , i , j in self.iter_region ( input ) :
            output [ i ] [ j ] = np.sum ( region * self.filters , axis = ( 1 , 2 ) )

        return output