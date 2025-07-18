module {
  func.func @main(%arg0: tensor<93x82xi64>, %arg1: tensor<35x96xi32>, %arg2: tensor<1x96xi32>, %arg3: tensor<98x92x58x38x100x97xi1>, %arg4: tensor<1x92x58x38x100x1xi1>) -> (tensor<35x96xi32>, tensor<98x92x58x38x100x97xi1>, tensor<93x1xi64>) {
    %0 = tosa.reduce_min %arg0 {axis = 1 : i32} : (tensor<93x82xi64>) -> tensor<93x1xi64>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<35x96xi32>, tensor<1x96xi32>) -> tensor<35x96xi32>
    %2 = tosa.logical_xor %arg3, %arg4 : (tensor<98x92x58x38x100x97xi1>, tensor<1x92x58x38x100x1xi1>) -> tensor<98x92x58x38x100x97xi1>
    %r_3 = tosa.const_shape {values = dense<[ 93, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.reshape %0, %r_3 : (tensor<93x1xi64>, !tosa.shape<2>) -> tensor<93x1xi64>
    return %1, %2, %3 : tensor<35x96xi32>, tensor<98x92x58x38x100x97xi1>, tensor<93x1xi64>
  }
}
