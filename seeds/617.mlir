module {
  func.func @main(%arg0: tensor<7x21x52x76x38x11xi64>, %arg1: tensor<7x21x52x76x26x11xi64>, %arg2: tensor<64x7x63x42x90x59xi1>, %arg3: tensor<1x1x63x1x90x59xi1>, %arg4: tensor<57x55x79x43xf32>, %arg5: tensor<57x55x79x43xf32>) -> (tensor<7x21x52x76x64x11xi64>, tensor<64x7x63x42x90x59xi1>, tensor<57x55x79x43xf32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 4 : i32} : (tensor<7x21x52x76x38x11xi64>, tensor<7x21x52x76x26x11xi64>) -> tensor<7x21x52x76x64x11xi64>
    %1 = tosa.logical_and %arg2, %arg3 : (tensor<64x7x63x42x90x59xi1>, tensor<1x1x63x1x90x59xi1>) -> tensor<64x7x63x42x90x59xi1>
    %2 = tosa.pow %arg4, %arg5 : (tensor<57x55x79x43xf32>, tensor<57x55x79x43xf32>) -> tensor<57x55x79x43xf32>
    return %0, %1, %2 : tensor<7x21x52x76x64x11xi64>, tensor<64x7x63x42x90x59xi1>, tensor<57x55x79x43xf32>
  }
}
