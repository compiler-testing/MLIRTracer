module {
  func.func @main(%arg0: tensor<61x59x35x21x64xi8>) -> tensor<61x59x35x21x64xi8> {
    %0 = tosa.clamp %arg0 {min_val = -16 : i8, max_val = 101 : i8} : (tensor<61x59x35x21x64xi8>) -> tensor<61x59x35x21x64xi8>
    return %0 : tensor<61x59x35x21x64xi8>
  }
}
