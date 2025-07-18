module {
  func.func @main(%arg0: tensor<3x31x64x49xi16>, %arg1: tensor<93x6x58x87xi8>, %arg2: tensor<93x6x58x1xi8>, %arg3: tensor<98x56x17xf32>, %arg4: tensor<98x56x17xf32>, %arg5: tensor<63x33x99x50x87x24xi1>, %arg6: tensor<63x1x1x1x1x1xi1>) -> (tensor<98x56x17xf32>, tensor<6x93x64x49xi16>, tensor<63x33x99x50x87x24xi1>, tensor<93x6x1x87xi8>) {
    %t_0 = tosa.const_shape {values = dense<[ 2, 3, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<3x31x64x49xi16>, !tosa.shape<4>) -> tensor<6x93x64x49xi16>
    %1 = tosa.maximum %arg1, %arg2 : (tensor<93x6x58x87xi8>, tensor<93x6x58x1xi8>) -> tensor<93x6x58x87xi8>
    %2 = tosa.pow %arg3, %arg4 : (tensor<98x56x17xf32>, tensor<98x56x17xf32>) -> tensor<98x56x17xf32>
    %3 = tosa.minimum %1, %1 : (tensor<93x6x58x87xi8>, tensor<93x6x58x87xi8>) -> tensor<93x6x58x87xi8>
    %4 = tosa.add %0, %0 : (tensor<6x93x64x49xi16>, tensor<6x93x64x49xi16>) -> tensor<6x93x64x49xi16>
    %5 = tosa.bitwise_and %3, %1 : (tensor<93x6x58x87xi8>, tensor<93x6x58x87xi8>) -> tensor<93x6x58x87xi8>
    %6 = tosa.logical_xor %arg5, %arg6 : (tensor<63x33x99x50x87x24xi1>, tensor<63x1x1x1x1x1xi1>) -> tensor<63x33x99x50x87x24xi1>
    %7 = tosa.reduce_max %5 {axis = 2 : i32} : (tensor<93x6x58x87xi8>) -> tensor<93x6x1x87xi8>
    return %2, %4, %6, %7 : tensor<98x56x17xf32>, tensor<6x93x64x49xi16>, tensor<63x33x99x50x87x24xi1>, tensor<93x6x1x87xi8>
  }
}
