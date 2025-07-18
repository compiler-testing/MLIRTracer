module {
  func.func @main(%arg0: tensor<49x78x73x14xi1>) -> tensor<49x78x73xi32> {
    %0 = tosa.reverse %arg0 {axis = 1 : i32} : (tensor<49x78x73x14xi1>) -> tensor<49x78x73x14xi1>
    %1 = tosa.argmax %0 {axis = 3 : i32} : (tensor<49x78x73x14xi1>) -> tensor<49x78x73xi32>
    return %1 : tensor<49x78x73xi32>
  }
}
