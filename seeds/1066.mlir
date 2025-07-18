module {
  func.func @main(%arg0: tensor<5x73x70x74x90x81xi16>, %arg1: tensor<5x73x81x74x90x81xi16>) -> tensor<5x73x151x74x90x81xi16> {
    %0 = tosa.concat %arg0, %arg1 {axis = 2 : i32} : (tensor<5x73x70x74x90x81xi16>, tensor<5x73x81x74x90x81xi16>) -> tensor<5x73x151x74x90x81xi16>
    return %0 : tensor<5x73x151x74x90x81xi16>
  }
}
