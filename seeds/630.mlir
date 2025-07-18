module {
  func.func @main(%arg0: tensor<24x37x70xi8>) -> tensor<24x1x70xi8> {
    %0 = tosa.reduce_product %arg0 {axis = 1 : i32} : (tensor<24x37x70xi8>) -> tensor<24x1x70xi8>
    return %0 : tensor<24x1x70xi8>
  }
}
