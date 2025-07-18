module {
  func.func @main(%arg0: tensor<94x33x9x2x82xi16>, %arg1: tensor<94x33x9x2x29xi16>, %arg2: tensor<64x60x66x61xi1>, %arg3: tensor<1x60x1x61xi1>) -> (tensor<94x33x9x2x111xi16>, tensor<64x60x66x61xi1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 4 : i32} : (tensor<94x33x9x2x82xi16>, tensor<94x33x9x2x29xi16>) -> tensor<94x33x9x2x111xi16>
    %1 = tosa.logical_and %arg2, %arg3 : (tensor<64x60x66x61xi1>, tensor<1x60x1x61xi1>) -> tensor<64x60x66x61xi1>
    return %0, %1 : tensor<94x33x9x2x111xi16>, tensor<64x60x66x61xi1>
  }
}
