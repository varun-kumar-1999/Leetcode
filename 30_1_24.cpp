#include<bits/stdc++.h>
using namespace std;
int main(){
ios_base::sync_with_stdio(false);
cin.tie(0);
cout.tie(0);
#ifndef ONLINE_JUDGE
    freopen("input.txt","r",stdin);
    freopen("output.txt","w",stdout);
#endif
int tc = 0;
cin>>tc;
while(tc--){
    int n,m;
    cin>>n>>m;
    cout<<n*(m/2)<<"\n";
}
return 0;
}