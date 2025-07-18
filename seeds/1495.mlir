module {
  func.func @main(%arg0: tensor<1x37x62x60x87x62xi8>) -> tensor<1x37x62x60x87x62xi8> {
    %0 = tosa.abs %arg0 : (tensor<1x37x62x60x87x62xi8>) -> tensor<1x37x62x60x87x62xi8>
    return %0 : tensor<1x37x62x60x87x62xi8>
  }
}
