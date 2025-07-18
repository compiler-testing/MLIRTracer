module {
  func.func @main(%arg0: tensor<42xi64>, %arg1: tensor<1xi64>, %arg2: tensor<58x93x17x89xf32>, %arg3: tensor<16x26x89x98xi32>, %arg4: tensor<16x1x89x1xi32>) -> (tensor<1xi1>, tensor<58x93x17x89xf32>, tensor<58x93x17x89xf32>, tensor<16x26x89x98xi32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<42xi64>, tensor<1xi64>) -> tensor<42xi1>
    %1 = tosa.tanh %arg2 : (tensor<58x93x17x89xf32>) -> tensor<58x93x17x89xf32>
    %2 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<42xi1>) -> tensor<1xi1>
    %3 = tosa.clz %2 : (tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.intdiv %arg3, %arg4 : (tensor<16x26x89x98xi32>, tensor<16x1x89x1xi32>) -> tensor<16x26x89x98xi32>
    %5 = tosa.exp %1 : (tensor<58x93x17x89xf32>) -> tensor<58x93x17x89xf32>
    %6 = tosa.ceil %5 : (tensor<58x93x17x89xf32>) -> tensor<58x93x17x89xf32>
    %7 = tosa.exp %1 : (tensor<58x93x17x89xf32>) -> tensor<58x93x17x89xf32>
    %8 = tosa.intdiv %4, %4 : (tensor<16x26x89x98xi32>, tensor<16x26x89x98xi32>) -> tensor<16x26x89x98xi32>
    return %3, %6, %7, %8 : tensor<1xi1>, tensor<58x93x17x89xf32>, tensor<58x93x17x89xf32>, tensor<16x26x89x98xi32>
  }
}
