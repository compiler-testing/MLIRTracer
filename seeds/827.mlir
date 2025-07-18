module {
  func.func @main(%arg0: tensor<76x6x91x56xi64>, %arg1: tensor<76x6x1x56xi64>, %arg2: tensor<65x21x82x27x29x17xf32>, %arg3: tensor<1x21x1x1x1x1xf32>) -> (tensor<65x21x82x27x29x17xi1>, tensor<65x21x82x27x29x17xf32>, tensor<76x6x91x56xi1>, tensor<1x6x91x56xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<76x6x91x56xi64>, tensor<76x6x1x56xi64>) -> tensor<76x6x91x56xi1>
    %1 = tosa.pow %arg2, %arg3 : (tensor<65x21x82x27x29x17xf32>, tensor<1x21x1x1x1x1xf32>) -> tensor<65x21x82x27x29x17xf32>
    %2 = tosa.greater_equal %1, %1 : (tensor<65x21x82x27x29x17xf32>, tensor<65x21x82x27x29x17xf32>) -> tensor<65x21x82x27x29x17xi1>
    %3 = tosa.tanh %1 : (tensor<65x21x82x27x29x17xf32>) -> tensor<65x21x82x27x29x17xf32>
    %4 = tosa.logical_not %0 : (tensor<76x6x91x56xi1>) -> tensor<76x6x91x56xi1>
    %5 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<76x6x91x56xi1>) -> tensor<1x6x91x56xi1>
    return %2, %3, %4, %5 : tensor<65x21x82x27x29x17xi1>, tensor<65x21x82x27x29x17xf32>, tensor<76x6x91x56xi1>, tensor<1x6x91x56xi1>
  }
}
