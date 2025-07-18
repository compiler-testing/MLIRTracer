module {
  func.func @main(%arg0: tensor<47x89x91x17xf32>) -> tensor<47x275366xf32> {
    %0 = tosa.rsqrt %arg0 : (tensor<47x89x91x17xf32>) -> tensor<47x89x91x17xf32>
    %1 = tosa.concat %0, %0 {axis = 3 : i32} : (tensor<47x89x91x17xf32>, tensor<47x89x91x17xf32>) -> tensor<47x89x91x34xf32>
    %2 = tosa.clamp %1 {min_val = -2.300000e+01 : f32, max_val = 7.600000e+01 : f32} : (tensor<47x89x91x34xf32>) -> tensor<47x89x91x34xf32>
    %r_3 = tosa.const_shape {values = dense<[ 47, 275366 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.reshape %2, %r_3 : (tensor<47x89x91x34xf32>, !tosa.shape<2>) -> tensor<47x275366xf32>
    return %3 : tensor<47x275366xf32>
  }
}
