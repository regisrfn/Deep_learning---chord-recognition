a=1
for i in *.wav; do
  new=$(printf "d-minor-%d.wav" "$a")
  mv -i "$i" "$new"
  let a=a+1
done