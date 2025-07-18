module {
  func.func @main(%arg0: tensor<17x19x60x99xi1>) -> tensor<17x19x60x99xi1> {
    %0 = tosa.reverse %arg0 {axis = 2 : i32} : (tensor<17x19x60x99xi1>) -> tensor<17x19x60x99xi1>
    return %0 : tensor<17x19x60x99xi1>
  }
}
