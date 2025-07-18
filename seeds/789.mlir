module {
  func.func @main(%arg0: tensor<94x100x60x7x37x38xi16>, %arg1: tensor<1x1x60x7x1x38xi16>, %arg2: tensor<60xi8>, %arg3: tensor<46x37x73x23x75xf32>) -> (tensor<94x100x60x7x37x76xi16>, tensor<1xi8>, tensor<46x37x73x23x75xf32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<94x100x60x7x37x38xi16>, tensor<1x1x60x7x1x38xi16>) -> tensor<94x100x60x7x37x38xi16>
    %1 = tosa.concat %0, %0 {axis = 5 : i32} : (tensor<94x100x60x7x37x38xi16>, tensor<94x100x60x7x37x38xi16>) -> tensor<94x100x60x7x37x76xi16>
    %2 = tosa.reverse %arg2 {axis = 0 : i32} : (tensor<60xi8>) -> tensor<60xi8>
    %3 = tosa.add %1, %1 : (tensor<94x100x60x7x37x76xi16>, tensor<94x100x60x7x37x76xi16>) -> tensor<94x100x60x7x37x76xi16>
    %4 = tosa.ceil %arg3 : (tensor<46x37x73x23x75xf32>) -> tensor<46x37x73x23x75xf32>
    %5 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<60xi8>) -> tensor<1xi8>
    %6 = tosa.log %4 : (tensor<46x37x73x23x75xf32>) -> tensor<46x37x73x23x75xf32>
    return %3, %5, %6 : tensor<94x100x60x7x37x76xi16>, tensor<1xi8>, tensor<46x37x73x23x75xf32>
  }
}
