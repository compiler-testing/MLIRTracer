module {
  func.func @main(%arg0: tensor<25x36x21x47xf32>) -> tensor<1x36x21x47xf32> {
    %0 = tosa.ceil %arg0 : (tensor<25x36x21x47xf32>) -> tensor<25x36x21x47xf32>
    %1 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<25x36x21x47xf32>) -> tensor<1x36x21x47xf32>
    return %1 : tensor<1x36x21x47xf32>
  }
}
