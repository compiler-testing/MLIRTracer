module {
  func.func @main(%arg0: tensor<58x27x84x17x58xi1>, %arg1: tensor<87x99x68x67x64xf32>) -> (tensor<58x27x84x17x58xi1>, tensor<87x99x68x67x64xf32>) {
    %0 = tosa.abs %arg0 : (tensor<58x27x84x17x58xi1>) -> tensor<58x27x84x17x58xi1>
    %1 = tosa.clz %0 : (tensor<58x27x84x17x58xi1>) -> tensor<58x27x84x17x58xi1>
    %2 = tosa.floor %arg1 : (tensor<87x99x68x67x64xf32>) -> tensor<87x99x68x67x64xf32>
    %3 = tosa.logical_not %1 : (tensor<58x27x84x17x58xi1>) -> tensor<58x27x84x17x58xi1>
    %4 = tosa.clamp %2 {min_val = -7.000000e+00 : f32, max_val = 7.700000e+01 : f32} : (tensor<87x99x68x67x64xf32>) -> tensor<87x99x68x67x64xf32>
    return %3, %4 : tensor<58x27x84x17x58xi1>, tensor<87x99x68x67x64xf32>
  }
}
