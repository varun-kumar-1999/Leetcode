#include<bits/stdc++.h>
using namespace std;
int solve(vector<int>& ar){
    sort(ar.begin(),ar.end());
   
    long long ans = -1,sum = 0;
    for(int i = 0;i<ar.size();i++){
        if(ar[i] < sum)
            ans = sum + ar[i];
        sum += ar[i];
    }

    return ans;
}
int main(){
ios_base::sync_with_stdio(false);
cin.tie(0);
cout.tie(0);
    //freopen("input.txt","r",stdin);
    freopen("output.txt","w",stdout);
vector<int> a = {1,12,1,2,5,50,3};
 int ans = solve(a);
cout<<ans;
return 0;
}