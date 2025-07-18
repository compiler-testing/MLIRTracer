module {
  func.func @main(%arg0: tensor<1x15x66xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<93x87x32x38xi8>, %arg3: tensor<71x6x55x50x40x33xi1>, %arg4: tensor<1x6x1x1x40x1xi1>) -> (tensor<1x15x66xf32>, tensor<93x1x32x38xi8>, tensor<71x6x55x50x40x33xi1>, tensor<1x15x66xf32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<1x15x66xf32>, tensor<1x1x1xf32>) -> tensor<1x15x66xf32>
    %1 = tosa.abs %0 : (tensor<1x15x66xf32>) -> tensor<1x15x66xf32>
    %2 = tosa.bitwise_not %arg2 : (tensor<93x87x32x38xi8>) -> tensor<93x87x32x38xi8>
    %3 = tosa.bitwise_or %2, %2 : (tensor<93x87x32x38xi8>, tensor<93x87x32x38xi8>) -> tensor<93x87x32x38xi8>
    %4 = tosa.reduce_product %3 {axis = 1 : i32} : (tensor<93x87x32x38xi8>) -> tensor<93x1x32x38xi8>
    %5 = tosa.logical_xor %arg3, %arg4 : (tensor<71x6x55x50x40x33xi1>, tensor<1x6x1x1x40x1xi1>) -> tensor<71x6x55x50x40x33xi1>
    %6 = tosa.log %0 : (tensor<1x15x66xf32>) -> tensor<1x15x66xf32>
    return %1, %4, %5, %6 : tensor<1x15x66xf32>, tensor<93x1x32x38xi8>, tensor<71x6x55x50x40x33xi1>, tensor<1x15x66xf32>
  }
}
