module {
  func.func @main(%arg0: tensor<27x99x72x83xf32>) -> tensor<27x99x72x166xf32> {
    %0 = tosa.log %arg0 : (tensor<27x99x72x83xf32>) -> tensor<27x99x72x83xf32>
    %1 = tosa.concat %0, %0 {axis = 3 : i32} : (tensor<27x99x72x83xf32>, tensor<27x99x72x83xf32>) -> tensor<27x99x72x166xf32>
    return %1 : tensor<27x99x72x166xf32>
  }
}
